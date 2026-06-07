# virt_knock — CUDA-Native Single-Cell Virtual Gene Knockout

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

GPU-accelerated implementation of the **scTenifoldKnk** algorithm (Osorio et al., *Patterns*, 2022) for predicting gene function via single-cell gene regulatory network perturbation.

> **~5× faster** than CPU-only implementations on real-world datasets.  
> Scales to **50,000+ cells** and **20,000+ genes** on a single GPU.  
> **v0.2.0** — Integrated enrichment analysis (GSEA + ORA), tqdm progress bars, timestamped outputs.

---

## How It Works

```
Expression matrix          PC networks (bootstrap)        Manifold alignment          Differential regulation       Enrichment analysis
(genes × cells)    ──►     (GPU randomized SVD)    ──►    (GPU Laplacian eigh)  ──►   (chi-square + FDR)    ──►   (GSEA + ORA)
        │                          │                              │                          │                        │
        ▼                          ▼                              ▼                          ▼                        ▼
   QC filtering            Tensor decomposition            WT vs KO embedding         Ranked gene list          GO / KEGG / Reactome
                          (bootstrap mean)                distance per gene           (p-value sorted)         pathway enrichment
```

The algorithm builds multiple gene regulatory networks from bootstrapped cell subsets, decomposes them into a wild-type tensor, simulates knockout by zeroing target gene rows, then measures the manifold embedding shift for every gene.

## Why CUDA?

The computational bottleneck is the **PC network construction**: for each of ~8,000 genes we regress it on all other genes via low-rank SVD. On a typical scRNA-seq matrix (8,000 genes × 42,000 cells), the core operation is:

```
M = Qᵀ @ X      (832 × 42,507) @ (42,507 × 7,901)  =  559 GFLOPs per batch
                                                         × 124 batches
                                                         ─────────────────
                                                         69 TFLOPs total
```

| Backend | Throughput (FP32) | Per-batch latency | PC network (est.) |
|---------|------------------:|------------------:|------------------:|
| **GPU (V100)** | **14.1 TFLOPS** | 39.5 ms | **22.5 s** |
| CPU (MKL, 12-core) | 496 GFLOPS | 1,128 ms | ~160 s |
| **Speedup** | — | **29×** | **~7×** |

> Benchmarked on TESLA V100-16GB (14.1 TFLOPS peak FP32) vs Intel Xeon (496 GFLOPS MKL).  
> Matrix: (832 × 42,507) @ (42,507 × 7,901) = 559 GFLOPs per batch.

## Installation

```bash
# Requires PyTorch with CUDA (see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install virt_knock with enrichment support
pip install "virt_knock[enrich] @ git+https://github.com/virchow-zhang/virt_knock.git"

# Or minimal install (no enrichment)
pip install git+https://github.com/virchow-zhang/virt_knock.git
```

Or from source:

```bash
git clone https://github.com/virchow-zhang/virt_knock.git
cd virt_knock
pip install -e ".[enrich]"
```

## Quick Start

### CLI

```bash
# Knock out ETS1 with default bootstrap (10 nets × 500 cells)
virt_knock -i expression.tsv -g ETS1

# Use all cells with multi-network averaging (recommended for robust results)
virt_knock -i expression.tsv -g ETS1 --no-bootstrap -n 10

# Knock out with automatic enrichment analysis (GSEA + ORA)
virt_knock -i expression.tsv -g ETS1 --no-bootstrap -n 10 --enrich

# Knock out multiple genes
virt_knock -i expression.tsv -g ETS1,FOXP3 -n 10 -c 500

# Custom output directory (timestamped by default)
virt_knock -i expression.tsv -g ETS1 -o ./results
```

> **Note**: Output directories are automatically timestamped (e.g., `output/ETS1_n10_20260607_120000/`).  
> Use `--no-timestamp` to disable.
> 
> **Subsampling warning**: When `-c ≤ 500`, the KO gene may not rank first due to insufficient signal.  
> Consider `--all-cells` or `--no-bootstrap -n 10` for biologically robust results.

### Python API

```python
import pandas as pd
from virt_knock import scTenifoldKnkCUDA

# Load expression matrix (genes × cells)
data = pd.read_csv("expression.tsv", sep="\t", index_col=0)

# Run virtual knockout
sc = scTenifoldKnkCUDA(
    data=data,
    ko_genes=["Ets1"],
    n_nets=10,          # bootstrap networks
    n_samp_cells=500,   # cells per subsample
    n_comp=3,           # PC components
    q=0.95,             # edge pruning threshold
)
result = sc.run()

# Top differentially regulated genes
print(result.head(20))
#         Gene  Distance       FC         p-value  adjusted_p-value
# 0       Ets1  0.000146  696711.44  0.000000e+00      0.000000e+00
# 1       Flt1  0.000003     372.41  5.578652e-83      2.203846e-79
# 2      Mcf2l  0.000003     288.74  9.341104e-65      2.460135e-61
# ...

# Save results
result.to_csv("knockout_results.csv")

# Run enrichment analysis (requires `pip install virt_knock[enrich]`)
from virt_knock import run_enrichment_all

run_enrichment_all(
    dreg_df=result,
    out_dir="./enrichment",
    ora_p_cutoff=0.05,  # adjusted p-value threshold for ORA
)
```

### Low-level API

For fine-grained control over each pipeline step:

```python
from virt_knock import (
    run_qc,
    build_pc_networks,
    tensor_decompose_mean,
    strict_direction,
    virtual_knockout,
    manifold_alignment,
    differential_regulation,
)

# 1. QC
qc_data = run_qc(data, min_lib_size=1000)

# 2. Build PC networks (GPU)
networks = build_pc_networks(qc_data, n_nets=10, n_samp_cells=500)

# 3. Tensor decomposition
wt_tensor = tensor_decompose_mean(networks, qc_data.index)

# 4. Knockout
ko_tensor = virtual_knockout(wt_tensor, ["Ets1"])

# 5. Manifold alignment (GPU)
ma = manifold_alignment(wt_tensor, ko_tensor, d=2)

# 6. Differential regulation
result = differential_regulation(ma, ko_genes=["Ets1"])

# 7. Enrichment analysis (optional, requires gseapy)
from virt_knock import run_gsea_prerank, run_ora_enrichr

# GSEA: rank all genes by Z-score
run_gsea_prerank(result, out_dir="./gsea")

# ORA: test significant genes against pathway databases
sig_genes = result[result["adjusted_p-value"] < 0.05]["Gene"].tolist()
run_ora_enrichr(sig_genes, out_dir="./ora")
```

## Algorithm Details

### PC Network Construction (GPU-accelerated)

For each gene *k*, we regress its expression vector **y** on all other genes **X₍₋ₖ₎** using **randomized low-rank SVD**:

**Key optimisation** — Instead of computing SVD of **X₍₋ₖ₎** for each gene independently (which would require ~8,000 expensive decompositions), we:

1. Precompute **XΩ** (random projection, one matmul)
2. For each gene *k*: compute the rank-1 update **X₍₋ₖ₎Ω = XΩ − outer(xₖ, ωₖ)**
3. Batch the QR decompositions (64 genes per batch) using `torch.linalg.qr`
4. Batch the back-projection **QᵀX** as a single large matmul

This reduces the per-gene cost from O(cells × genes) to O(cells × rank + genes × rank²).

### Benchmark

```
Benchmark (TESLA V100-16GB, 42,507 cells × 7,901 genes)
═══════════════════════════════════════════════════════════

  Matmul core:                  GPU  39.5 ms  (14.1 TFLOPS)
                                CPU 1128  ms  ( 496 GFLOPS)
                                ═══════════════════════════
                                GPU 29× faster

  PC network (full pipeline):   GPU   22.5 s
                                CPU  ~160  s  (estimated)
                                ═══════════════════════════
                                GPU  7× faster

  virt_knock end-to-end:        GPU   47 s   (all steps)
                                CPU  ~190 s  (estimated)
                                ═══════════════════════════
                                GPU  4× faster
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | *(required)* | Expression matrix (tsv/csv, genes × cells) |
| `-g`, `--genes` | *(required)* | Gene name(s) to knock out (comma-separated) |
| `-o`, `--output` | `./output` | Base output directory (timestamped subdir created) |
| `--no-timestamp` | `false` | Disable timestamped subdirectory |
| `--enrich` | `false` | Run enrichment analysis (GSEA + ORA) after knockout |
| `-n`, `--n-nets` | `10` | Number of bootstrap networks |
| `-c`, `--n-cells` | `500` | Cells per subsample |
| `--all-cells` | `false` | Use all cells (equivalent to `--no-bootstrap` with `-n 1`) |
| `--no-bootstrap` | `false` | Use all cells in each network (no cell subsampling) |
| `--n-comp` | `3` | PC components per gene regression |
| `--q` | `0.95` | Quantile threshold for edge pruning |
| `--strict-lambda` | `0` | Direction pruning strength (0 = off) |
| `--ma-dim` | `2` | Manifold alignment dimension |
| `--seed` | `42` | Random seed |
| `--min-lib-size` | `1000` | Min library size per cell |
| `--min-percent` | `0.05` | Min fraction of cells expressing a gene |
| `--sep` | `tab` | Field separator in input file |

## Output Structure

```
output/Ets1_n10_20260607_120000/
├── knockout_Ets1_d_regulation.csv     # Differential regulation (7 columns: Gene, Distance, Z, FC, p-value, adjusted_p-value)
└── enrichment/                          # Only when --enrich is used
    ├── gsea/
    │   ├── gsea_GO_Biological_Process_2023/
    │   ├── gsea_KEGG_2019_Mouse/
    │   └── gsea_Reactome_2022/
    └── ora/
        ├── ora_GO_Biological_Process_2023/
        ├── ora_KEGG_2019_Mouse/
        └── ora_Reactome_2022/
```

## Input Format

Expression matrix as a **tab-separated** file:

| Gene | Cell_1 | Cell_2 | ... | Cell_N |
|------|--------|--------|-----|--------|
| Xkr4 | 0.0 | 0.0 | ... | 1.2 |
| Ets1 | 3.5 | 2.1 | ... | 0.0 |
| ... | ... | ... | ... | ... |

- First column: gene names (unique)
- First row: cell barcodes
- Values: raw or normalised expression counts

## Citation

If you use virt_knock, please cite:

- **Method**: Osorio, D. et al. (2022). *scTenifoldKnk: An efficient virtual knockout tool for gene function predictions via single-cell gene regulatory network perturbation.* Patterns, 3(3), 100434. [DOI: 10.1016/j.patter.2022.100434](https://doi.org/10.1016/j.patter.2022.100434)
- **Software**: virt_knock (this repository)

## License

MIT — see [LICENSE](LICENSE) for details.
