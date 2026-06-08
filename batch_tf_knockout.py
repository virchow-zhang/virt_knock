#!/usr/bin/env python3
"""Batch TF virtual knockout pipeline.

Runs QC once, builds shared PC networks & WT tensor, then loops over all
QC-passing TFs: virtual knockout → manifold alignment → differential
regulation → (optional) enrichment (GSEA + ORA).

Resume-friendly: skips TFs already processed.
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from virt_knock import (
    run_qc,
    build_pc_networks,
    tensor_decompose_mean,
    strict_direction,
    virtual_knockout,
    manifold_alignment,
    differential_regulation,
    run_enrichment_all,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MATRIX_PATH = r"D:\OneDrive\bioinformatics\01.TAC\fxgsys-1p\virtual_knockout_cuda\expression_celltype_Cardiomyocytes.txt"
TF_LIST_PATH = r"D:\OneDrive\bioinformatics\01.TAC\fxgsys-1p\virtual_knockout_cuda\tf_list.txt"
OUTPUT_DIR = r"D:\OneDrive\bioinformatics\01.TAC\fxgsys-1p\virtual_knockout_cuda\output_tf_batch"

# Pipeline parameters
N_NETS = 10
N_SAMP_CELLS = 500  # cells per bootstrap sample; None = all cells
N_COMP = 3
Q_THRESH = 0.95
STRICT_LAMBDA = 0.0
MA_DIM = 2
SEED = 42

# QC parameters
QC_KWS = {
    "min_lib_size": 1000,
    "remove_outlier_cells": True,
    "min_percent": 0.05,
    "min_exp_avg": 0.05,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CLI arguments ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Batch TF virtual knockout pipeline",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python batch_tf_knockout.py                      # run with enrichment (default)
  python batch_tf_knockout.py --no-enrich          # skip enrichment for speed
  python batch_tf_knockout.py -c 1000 -n 20        # custom bootstrap params
""",
)
parser.add_argument("--no-enrich", dest="enrich", action="store_false", default=True,
                    help="Skip enrichment analysis (GSEA + ORA)")
parser.add_argument("-c", "--n-cells", dest="n_samp_cells", type=int, default=N_SAMP_CELLS,
                    help=f"Cells per bootstrap sample (default: {N_SAMP_CELLS})")
parser.add_argument("-n", "--n-nets", dest="n_nets", type=int, default=N_NETS,
                    help=f"Number of bootstrap networks (default: {N_NETS})")
parser.add_argument("--n-comp", dest="n_comp", type=int, default=N_COMP,
                    help=f"PC components (default: {N_COMP})")
args = parser.parse_args()

DO_ENRICH = args.enrich
N_NETS = args.n_nets
N_SAMP_CELLS = args.n_samp_cells
N_COMP = args.n_comp


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & QC
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  Batch TF Virtual Knockout Pipeline")
print("=" * 60)
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Output:  {OUTPUT_DIR}")
print(f"  Enrichment: {'ON' if DO_ENRICH else 'OFF'}")

print("\n[1] Loading expression matrix ...")
data = pd.read_csv(MATRIX_PATH, sep="\t", index_col=0)
print(f"    Loaded: {data.shape[0]:,} genes x {data.shape[1]:,} cells")

print("\n[2] Running QC ...")
qc_data = run_qc(data, **QC_KWS)
print(f"    QC result: {qc_data.shape[0]:,} genes x {qc_data.shape[1]:,} cells")
del data

qc_matrix_path = os.path.join(OUTPUT_DIR, "qc_expression_matrix.csv.gz")
qc_data.to_csv(qc_matrix_path, compression="gzip")
print(f"    Saved QC matrix: {qc_matrix_path}")

# ── Identify QC-passing TFs ──────────────────────────────────────────────────

with open(TF_LIST_PATH) as f:
    all_tfs = sorted(set(line.strip() for line in f if line.strip()))

# Case-insensitive match to QC genes
idx_lower = {g.lower(): g for g in qc_data.index}
qc_tfs = []
for tf in all_tfs:
    match = idx_lower.get(tf.lower())
    if match:
        qc_tfs.append(match)

print(f"\n    TFs in input list:   {len(all_tfs)}")
print(f"    TFs passing QC:      {len(qc_tfs)}")
with open(os.path.join(OUTPUT_DIR, "qc_tf_list.txt"), "w") as f:
    for t in qc_tfs:
        f.write(t + "\n")

if not qc_tfs:
    print("ERROR: No TFs passed QC. Exiting.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Shared PC networks & WT tensor (computed ONCE)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Building PC networks (shared across all TFs) ...")
print(f"    n_nets={N_NETS}, n_samp_cells={N_SAMP_CELLS}, n_comp={N_COMP}")

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
print(f"    Built {len(networks)} networks in {elapsed:.1f}s")

print("\n[4] Building WT tensor ...")
t0 = time.perf_counter()
wt_tensor = tensor_decompose_mean(networks, gene_names)
del networks

wt_arr = strict_direction(wt_tensor.values, STRICT_LAMBDA).T.copy()
np.fill_diagonal(wt_arr, 0)
wt_tensor = pd.DataFrame(wt_arr, index=wt_tensor.index, columns=wt_tensor.columns)

nnz = (wt_tensor.values != 0).sum()
print(f"    WT tensor: {wt_tensor.shape}, non-zero edges: {nnz:,}")
print(f"    Wall time: {time.perf_counter() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Batch KO loop
# ═══════════════════════════════════════════════════════════════════════════════

# Resume: determine already-done TFs
done = set()
for d in os.listdir(OUTPUT_DIR):
    p = os.path.join(OUTPUT_DIR, d, f"knockout_{d}_d_regulation.csv")
    if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and os.path.exists(p):
        done.add(d)

pending = [tf for tf in qc_tfs if tf not in done]
print(f"\n    Already done: {len(done)}  |  Pending: {len(pending)}")

results_summary = {}
failed = []
log_lines = []

t_start = time.perf_counter()

for i, tf in enumerate(pending):
    tf_dir = os.path.join(OUTPUT_DIR, f"{tf}_KO")
    print(f"\n{'=' * 60}")
    print(f"  [{i+1}/{len(pending)}] KO: {tf}")
    print(f"{'=' * 60}")

    try:
        # ── 5. Virtual knockout ──────────────────────────────────────────
        t0 = time.perf_counter()
        ko_tensor = virtual_knockout(wt_tensor, [tf])
        print(f"    [KO] {time.perf_counter() - t0:.1f}s")

        # ── 6. Manifold alignment ────────────────────────────────────────
        t0 = time.perf_counter()
        ma_result = manifold_alignment(wt_tensor, ko_tensor, d=MA_DIM)
        print(f"    [MA] {time.perf_counter() - t0:.1f}s")

        # ── 7. Differential regulation ───────────────────────────────────
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
            f"{'OK' if not DO_ENRICH else 'OK+ENRICH'}"
        )
        log_lines.append(log_line)
        print(f"    Sig genes: {n_sig}/{len(d_reg)} (adj.p < 0.05)")

        # ── 8. Enrichment (optional) ─────────────────────────────────────
        if DO_ENRICH:
            enrich_dir = os.path.join(tf_dir, "enrichment")
            run_enrichment_all(
                dreg_df=d_reg,
                out_dir=enrich_dir,
                verbose=True,
            )
        del d_reg

    except Exception as exc:
        print(f"    FAILED: {exc}")
        traceback.print_exc()
        failed.append(tf)
        log_lines.append(f"{tf}\t0\t0\tN/A\tN/A\tFAILED:{exc}")

    # Periodic save of log
    if (i + 1) % 5 == 0:
        with open(os.path.join(OUTPUT_DIR, "batch_log.tsv"), "w") as f:
            f.write("TF\tn_sig\tn_total\ttop_gene\ttop_pval\tstatus\n")
            f.write("\n".join(log_lines) + "\n")

total_elapsed = time.perf_counter() - t_start

# ═══════════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print(f"  BATCH COMPLETE")
print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Processed:  {len(pending) - len(failed)}/{len(pending)}")
print(f"  Total time: {total_elapsed/3600:.1f}h ({total_elapsed:.0f}s)")
if failed:
    print(f"  Failed TFs: {failed}")
print(f"{'=' * 60}")

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
