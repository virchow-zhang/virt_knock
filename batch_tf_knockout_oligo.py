#!/usr/bin/env python3
"""Oligodendrocytes Batch TF Virtual Knockout Pipeline
======================================================
Loads the full Oligodendrocytes expression matrix, identifies all TFs from
TRRUST + ENCODE/ChEA consensus databases, runs QC, builds 10 PC networks
on ALL cells (no subsampling), then performs virtual knockout on every
QC-passing TF with enrichment analysis (GSEA + ORA).

Reference: Osorio et al., Patterns, 2022 (scTenifoldKnk)
"""

import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, r"D:\OneDrive\bioinformatics\01.TAC\fxgsys-1p\virtual_knockout_cuda")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import numpy as np
import pandas as pd
import gseapy
import torch

from virt_knock.core import (
    run_qc,
    build_pc_networks,
    tensor_decompose_mean,
    strict_direction,
    virtual_knockout,
    manifold_alignment,
    differential_regulation,
)
from virt_knock.enrichment import run_enrichment_all

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

MATRIX_PATH = r"D:\OneDrive\bioinformatics\MCAO\GSE331114\expression_matrices\expr_Oligodendrocytes.csv.gz"
OUTPUT_DIR = r"D:\OneDrive\bioinformatics\MCAO\GSE331114\expression_matrices\output_oligo_tf_batch_v2"

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline parameters
# ═══════════════════════════════════════════════════════════════════════════════

N_NETS = 10               # 10 PC network convergence iterations
N_SAMP_CELLS = None        # ALL cells (no subsampling, not 500!)
N_COMP = 3                 # PC components per gene regression
Q_THRESH = 0.95            # Quantile threshold for edge pruning
STRICT_LAMBDA = 0.0
MA_DIM = 2                 # Manifold alignment dimension
SEED = 42

# QC parameters
QC_KWS = {
    "min_lib_size": 1000,
    "remove_outlier_cells": True,
    "min_percent": 0.01,      # 1%表达率 —— 平衡基因数与显存
    "min_exp_avg": 0.01,
}

# ── Enrichment: use HUMAN libraries (this is human brain data) ────────────────
GSEA_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
]
ORA_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load expression matrix
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Oligodendrocytes Batch TF Virtual Knockout Pipeline")
print("  scTenifoldKnk - CUDA Native")
print("=" * 70)
print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Output:     {OUTPUT_DIR}")
print(f"  N nets:     {N_NETS}")
print(f"  N samp:     ALL CELLS (no subsampling)")
print(f"  Enrichment: ON (GSEA + ORA)")

print("\n[1] Loading expression matrix ...")
data = pd.read_csv(MATRIX_PATH, index_col=0)
print(f"    Loaded: {data.shape[0]:,} genes x {data.shape[1]:,} cells")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. QC
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Running QC ...")
qc_data = run_qc(data, **QC_KWS)
print(f"    QC result: {qc_data.shape[0]:,} genes x {qc_data.shape[1]:,} cells")
del data

qc_matrix_path = os.path.join(OUTPUT_DIR, "qc_expression_matrix.csv.gz")
qc_data.to_csv(qc_matrix_path, compression="gzip")
print(f"    Saved QC matrix: {qc_matrix_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Identify TFs from matrix using TRRUST + ENCODE/ChEA consensus
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Identifying transcription factors from QC-passed genes ...")

tf_set = set()

# TRRUST Transcription Factors 2019
try:
    trrust = gseapy.parser.get_library(name="TRRUST_Transcription_Factors_2019")
    for key in trrust:
        # Keys are like "ETS1 human", "FOS mouse"
        tf_name = key.rsplit(" ", 1)[0].upper()
        tf_set.add(tf_name)
    print(f"    TRRUST: {len(trrust)} entries → {len(tf_set)} unique TF symbols")
except Exception as e:
    print(f"    TRRUST download failed: {e}")

# ENCODE + ChEA Consensus TFs from ChIP-X
try:
    encode_chea = gseapy.parser.get_library(name="ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X")
    for key in encode_chea:
        # Keys are like "ETS1 ENCODE", "NANOG CHEA"
        tf_name = key.split(" ")[0].upper()
        tf_set.add(tf_name)
    print(f"    ENCODE+ChEA: {len(encode_chea)} entries → {len(tf_set)} total unique TFs")
except Exception as e:
    print(f"    ENCODE+ChEA download failed: {e}")

# ENCODE TF ChIP-seq 2015 (supplementary)
try:
    encode = gseapy.parser.get_library(name="ENCODE_TF_ChIP-seq_2015")
    for key in encode:
        tf_name = key.split(" ")[0].upper()
        tf_set.add(tf_name)
    print(f"    ENCODE 2015: {len(encode)} entries → {len(tf_set)} total unique TFs")
except Exception as e:
    print(f"    ENCODE 2015 download failed: {e}")

print(f"    Combined TF set: {len(tf_set)} unique symbols")

# Cross-reference with QC-passed genes (case-insensitive)
qc_idx_upper = {g.upper(): g for g in qc_data.index}
qc_tfs = []
for tf in sorted(tf_set):
    match = qc_idx_upper.get(tf)
    if match:
        qc_tfs.append(match)

print(f"    TFs in combined set:    {len(tf_set)}")
print(f"    TFs passing QC in matrix: {len(qc_tfs)}")

with open(os.path.join(OUTPUT_DIR, "qc_tf_list.txt"), "w") as f:
    for t in qc_tfs:
        f.write(t + "\n")

if not qc_tfs:
    print("ERROR: No TFs passed QC. Exiting.")
    sys.exit(1)

print(f"    First 30 TFs: {qc_tfs[:30]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Build shared PC networks & WT tensor (computed ONCE, skip if cached)
# ═══════════════════════════════════════════════════════════════════════════════

wt_path = os.path.join(OUTPUT_DIR, "WT_tensor.csv.gz")

if os.path.exists(wt_path):
    print(f"\n[4/5] WT tensor already cached, loading ...")
    wt_tensor = pd.read_csv(wt_path, index_col=0)
    print(f"    Loaded WT tensor: {wt_tensor.shape}")
else:
    print(f"\n[4] Building {N_NETS} PC networks (ALL cells, no subsampling) ...")
    print(f"    n_genes={qc_data.shape[0]}, n_cells={qc_data.shape[1]}, n_comp={N_COMP}")
    print(f"    GPU memory estimate: ~{qc_data.shape[0]*qc_data.shape[1]*4/1e9:.1f} GB (X)")
    print(f"    Adjacency matrix: ~{qc_data.shape[0]**2*4/1e9:.1f} GB (A)")

    gene_names = qc_data.index.to_numpy()
    t0 = time.perf_counter()
    networks = build_pc_networks(
        qc_data,
        n_nets=N_NETS,
        n_samp_cells=N_SAMP_CELLS,
        n_comp=N_COMP,
        q=Q_THRESH,
        random_state=SEED,
    )
    elapsed = time.perf_counter() - t0
    print(f"    Built {len(networks)} networks in {elapsed:.1f}s ({elapsed/3600:.2f}h)")

    print("\n[5] Building WT tensor (mean of 10 networks) ...")
    t0 = time.perf_counter()
    wt_tensor = tensor_decompose_mean(networks, gene_names)
    del networks

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except (RuntimeError, Exception):
            pass
        try:
            print(f"    GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
        except Exception:
            pass

    wt_arr = strict_direction(wt_tensor.values, STRICT_LAMBDA).T.copy()
    np.fill_diagonal(wt_arr, 0)
    wt_tensor = pd.DataFrame(wt_arr, index=wt_tensor.index, columns=wt_tensor.columns)

    nnz = (wt_tensor.values != 0).sum()
    print(f"    WT tensor: {wt_tensor.shape}, non-zero edges: {nnz:,}")
    print(f"    Wall time: {time.perf_counter() - t0:.1f}s")

    wt_tensor.to_csv(wt_path, compression="gzip")
    print(f"    Saved: {wt_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Batch KO loop
# ═══════════════════════════════════════════════════════════════════════════════

# Resume-friendly: determine already-done TFs
done = set()
for d in os.listdir(OUTPUT_DIR):
    reg_path = os.path.join(OUTPUT_DIR, d, f"knockout_{d}_d_regulation.csv")
    if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and os.path.exists(reg_path):
        done.add(d)

pending = [tf for tf in qc_tfs if tf not in done]
print(f"\n    Already done: {len(done)}  |  Pending: {len(pending)}")

results_summary = {}
failed = []
log_lines = []

t_start = time.perf_counter()

for i, tf in enumerate(pending):
    tf_dir = os.path.join(OUTPUT_DIR, tf)
    print(f"\n{'=' * 70}")
    print(f"  [{i+1}/{len(pending)}] KO: {tf}")
    print(f"{'=' * 70}")

    try:
        # ── Virtual Knockout ────────────────────────────────────────────
        t0 = time.perf_counter()
        ko_tensor = virtual_knockout(wt_tensor, [tf])
        print(f"    [KO] {time.perf_counter() - t0:.1f}s")

        # ── Manifold Alignment (GPU eigh) ───────────────────────────────
        t0 = time.perf_counter()
        ma_result = manifold_alignment(wt_tensor, ko_tensor, d=MA_DIM)
        print(f"    [MA] {time.perf_counter() - t0:.1f}s")

        # ── Differential Regulation ───────────────────────────────────
        t0 = time.perf_counter()
        d_reg = differential_regulation(ma_result, n_ko_genes=1, ko_genes=[tf])
        print(f"    [DR] {time.perf_counter() - t0:.1f}s")
        del ma_result

        # Save d_reg
        os.makedirs(tf_dir, exist_ok=True)
        reg_path = os.path.join(tf_dir, f"knockout_{tf}_d_regulation.csv")
        d_reg.to_csv(reg_path, index=False)

        n_sig = (d_reg["adjusted_p-value"] < 0.05).sum()
        results_summary[tf] = {"n_sig": n_sig, "n_total": len(d_reg)}

        log_line = (
            f"{tf}\t{n_sig}\t{len(d_reg)}\t"
            f"{d_reg.iloc[0]['Gene']}\t{d_reg.iloc[0]['p-value']:.2e}\t"
            f"OK+ENRICH"
        )
        log_lines.append(log_line)
        print(f"    Sig genes: {n_sig}/{len(d_reg)} (adj.p < 0.05)")

        # ── Enrichment (GSEA + ORA) ────────────────────────────────────
        enrich_dir = os.path.join(tf_dir, "enrichment")
        run_enrichment_all(
            dreg_df=d_reg,
            out_dir=enrich_dir,
            gsea_libraries=GSEA_LIBRARIES,
            ora_libraries=ORA_LIBRARIES,
            verbose=True,
        )
        del d_reg

    except Exception as exc:
        print(f"    FAILED: {exc}")
        traceback.print_exc()
        failed.append(tf)
        log_lines.append(f"{tf}\t0\t0\tN/A\tN/A\tFAILED:{exc}")

    # Periodic save of log (every 5 TFs)
    if (i + 1) % 5 == 0:
        with open(os.path.join(OUTPUT_DIR, "batch_log.tsv"), "w") as f:
            f.write("TF\tn_sig\tn_total\ttop_gene\ttop_pval\tstatus\n")
            f.write("\n".join(log_lines) + "\n")
        print(f"    [Log saved at {i+1}/{len(pending)}]")

total_elapsed = time.perf_counter() - t_start

# ═══════════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  BATCH COMPLETE")
print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Processed:  {len(pending) - len(failed)}/{len(pending)}")
print(f"  Total KO time: {total_elapsed/3600:.1f}h ({total_elapsed:.0f}s)")
if failed:
    print(f"  Failed TFs: {failed}")
print(f"{'=' * 70}")

# Save final log
with open(os.path.join(OUTPUT_DIR, "batch_log.tsv"), "w") as f:
    f.write("TF\tn_sig\tn_total\ttop_gene\ttop_pval\tstatus\n")
    f.write("\n".join(log_lines) + "\n")

# Summary stats
with open(os.path.join(OUTPUT_DIR, "batch_summary.tsv"), "w") as f:
    f.write("TF\tn_sig\tn_total\n")
    for tf, info in sorted(results_summary.items()):
        f.write(f"{tf}\t{info['n_sig']}\t{info['n_total']}\n")

print(f"\n  Log saved: {os.path.join(OUTPUT_DIR, 'batch_log.tsv')}")
print(f"  Summary:   {os.path.join(OUTPUT_DIR, 'batch_summary.tsv')}")
print(f"\n  Done.")
