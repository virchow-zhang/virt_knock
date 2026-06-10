#!/usr/bin/env python3
"""Batch TF virtual knockout pipeline.

Runs QC once, builds shared PC networks & WT tensor (cached), then loops over
all QC-passing TFs: virtual knockout → manifold alignment → differential
regulation → enrichment (GSEA + ORA).

Resume-friendly: skips TFs already processed.

Examples:
  # Use a TF list file
  python examples/batch_tf_knockout.py \\
      --matrix expr_matrix.csv.gz \\
      --output ./output_batch \\
      --tf-list tf_list.txt

  # Auto-download TFs from Enrichr libraries
  python examples/batch_tf_knockout.py \\
      --matrix expr_matrix.csv.gz \\
      --output ./output_batch \\
      --tf-source TRRUST_Transcription_Factors_2019 ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X
"""

import argparse
import gc
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch TF virtual knockout pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/batch_tf_knockout.py -m expr.csv.gz -o ./out -l tfs.txt
  python examples/batch_tf_knockout.py -m expr.csv.gz -o ./out --no-enrich
  python examples/batch_tf_knockout.py -m expr.csv.gz -o ./out \\
      --tf-source TRRUST_Transcription_Factors_2019
        """,
    )
    p.add_argument("-m", "--matrix", required=True,
                   help="Path to expression matrix CSV/CSV.GZ (genes x cells)")
    p.add_argument("-o", "--output", required=True,
                   help="Output directory")
    p.add_argument("-l", "--tf-list",
                   help="Path to TF list file (one gene symbol per line). "
                        "If omitted, TFs are discovered via --tf-source.")
    p.add_argument("--tf-source", nargs="+",
                   default=["TRRUST_Transcription_Factors_2019",
                            "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X",
                            "ENCODE_TF_ChIP-seq_2015"],
                   help="Enrichr library names for TF discovery (default: TRRUST + ENCODE/ChEA + ENCODE 2015)")
    p.add_argument("--no-enrich", dest="enrich", action="store_false", default=True,
                   help="Skip enrichment analysis")
    p.add_argument("--gsea-libs", nargs="+",
                   default=["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"],
                   help="GSEA library names (default: GO_BP + KEGG_Human + Reactome)")
    p.add_argument("--ora-libs", nargs="+",
                   default=["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"],
                   help="ORA library names")
    p.add_argument("-n", "--n-nets", type=int, default=10,
                   help="Number of bootstrap PC networks (default: 10)")
    p.add_argument("-s", "--n-samp-cells", type=int, default=None,
                   help="Cells per subsample; omit for all cells (default)")
    p.add_argument("--n-comp", type=int, default=3,
                   help="PC components per gene regression (default: 3)")
    p.add_argument("--q-thresh", type=float, default=0.95,
                   help="Quantile threshold for edge pruning (default: 0.95)")
    p.add_argument("--min-lib-size", type=float, default=1000,
                   help="Min library size per cell (default: 1000)")
    p.add_argument("--min-percent", type=float, default=0.01,
                   help="Min fraction of cells expressing a gene (default: 0.01)")
    p.add_argument("--min-exp-avg", type=float, default=0.01,
                   help="Min mean expression (default: 0.01)")
    p.add_argument("--ma-dim", type=int, default=2,
                   help="Manifold alignment dimension (default: 2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-resume", action="store_true",
                   help="Force re-run all TFs (ignore cached results)")
    p.add_argument("--force-wt", action="store_true",
                   help="Force rebuild WT tensor (ignore cache)")
    return p.parse_args()


def discover_tfs_from_enrichr(library_names, qc_genes):
    """Download TF gene sets from Enrichr and cross-reference with QC genes."""
    import gseapy

    tf_set = set()
    for lib_name in library_names:
        try:
            lib = gseapy.parser.get_library(name=lib_name)
            for key in lib:
                tf_name = key.split(" ")[0].upper()
                tf_set.add(tf_name)
            print(f"    {lib_name}: {len(lib)} entries → {len(tf_set)} unique TFs")
        except Exception as e:
            print(f"    {lib_name} download failed: {e}")

    qc_upper = {g.upper(): g for g in qc_genes}
    qc_tfs = sorted(qc_upper[tf] for tf in sorted(tf_set) if tf in qc_upper)
    return qc_tfs


def load_tfs_from_file(tf_path, qc_genes):
    """Load TF list from file and cross-reference with QC genes."""
    with open(tf_path) as f:
        all_tfs = sorted(set(line.strip() for line in f if line.strip()))
    qc_lower = {g.lower(): g for g in qc_genes}
    qc_tfs = [qc_lower[tf.lower()] for tf in all_tfs if tf.lower() in qc_lower]
    return qc_tfs


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ── Header ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Batch TF Virtual Knockout Pipeline")
    print("  scTenifoldKnk - CUDA Native")
    print("=" * 70)
    print(f"  Started:     {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Matrix:      {args.matrix}")
    print(f"  Output:      {args.output}")
    print(f"  N nets:      {args.n_nets}")
    print(f"  N samp:      {'ALL' if args.n_samp_cells is None else args.n_samp_cells}")
    print(f"  Enrichment:  {'ON' if args.enrich else 'OFF'}")

    # ── 1. Load ───────────────────────────────────────────────────────────
    print("\n[1] Loading expression matrix ...")
    data = pd.read_csv(args.matrix, index_col=0)
    if data.shape[1] > data.shape[0]:
        data = data.T
    print(f"    Loaded: {data.shape[0]:,} genes x {data.shape[1]:,} cells")

    # ── 2. QC ─────────────────────────────────────────────────────────────
    print("\n[2] Running QC ...")
    qc_kws = {
        "min_lib_size": args.min_lib_size,
        "remove_outlier_cells": True,
        "min_percent": args.min_percent,
        "min_exp_avg": args.min_exp_avg,
    }
    qc_data = run_qc(data, **qc_kws)
    print(f"    QC result: {qc_data.shape[0]:,} genes x {qc_data.shape[1]:,} cells")
    del data

    qc_matrix_path = os.path.join(args.output, "qc_expression_matrix.csv.gz")
    qc_data.to_csv(qc_matrix_path, compression="gzip")
    print(f"    Saved QC matrix: {qc_matrix_path}")

    # ── 3. Identify TFs ───────────────────────────────────────────────────
    print("\n[3] Identifying transcription factors ...")

    if args.tf_list:
        print(f"    Loading TFs from: {args.tf_list}")
        qc_tfs = load_tfs_from_file(args.tf_list, qc_data.index)
    else:
        print(f"    Discovering TFs from Enrichr: {args.tf_source}")
        qc_tfs = discover_tfs_from_enrichr(args.tf_source, qc_data.index)

    print(f"    TFs passing QC in matrix: {len(qc_tfs)}")

    with open(os.path.join(args.output, "qc_tf_list.txt"), "w") as f:
        for t in qc_tfs:
            f.write(t + "\n")

    if not qc_tfs:
        print("ERROR: No TFs passed QC. Exiting.")
        sys.exit(1)

    print(f"    First 30 TFs: {qc_tfs[:30]}")

    # ── 4/5. Build WT tensor (cached) ─────────────────────────────────────
    wt_path = os.path.join(args.output, "WT_tensor.csv.gz")

    if os.path.exists(wt_path) and not args.force_wt:
        print(f"\n[4/5] WT tensor already cached, loading ...")
        wt_tensor = pd.read_csv(wt_path, index_col=0)
        print(f"    Loaded WT tensor: {wt_tensor.shape}")
    else:
        print(f"\n[4] Building {args.n_nets} PC networks ...")
        print(f"    n_genes={qc_data.shape[0]}, n_cells={qc_data.shape[1]}, n_comp={args.n_comp}")

        gene_names = qc_data.index.to_numpy()
        t0 = time.perf_counter()
        networks = build_pc_networks(
            qc_data,
            n_nets=args.n_nets,
            n_samp_cells=args.n_samp_cells,
            n_comp=args.n_comp,
            q=args.q_thresh,
            random_state=args.seed,
        )
        elapsed = time.perf_counter() - t0
        print(f"    Built {len(networks)} networks in {elapsed:.1f}s ({elapsed/3600:.2f}h)")

        print("\n[5] Building WT tensor (mean of networks) ...")
        t0 = time.perf_counter()
        wt_tensor = tensor_decompose_mean(networks, gene_names)
        del networks

        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            try:
                print(f"    GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, "
                      f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
            except Exception:
                pass

        wt_arr = strict_direction(wt_tensor.values, 0.0).T.copy()
        np.fill_diagonal(wt_arr, 0)
        wt_tensor = pd.DataFrame(wt_arr, index=wt_tensor.index, columns=wt_tensor.columns)

        nnz = (wt_tensor.values != 0).sum()
        print(f"    WT tensor: {wt_tensor.shape}, non-zero edges: {nnz:,}")
        print(f"    Wall time: {time.perf_counter() - t0:.1f}s")

        wt_tensor.to_csv(wt_path, compression="gzip")
        print(f"    Saved: {wt_path}")

    # ── 5. Batch KO loop ──────────────────────────────────────────────────
    if args.no_resume:
        done = set()
    else:
        done = set()
        for d in os.listdir(args.output):
            reg_path = os.path.join(args.output, d, f"knockout_{d}_d_regulation.csv")
            if os.path.isdir(os.path.join(args.output, d)) and os.path.exists(reg_path):
                done.add(d)

    pending = [tf for tf in qc_tfs if tf not in done]
    print(f"\n    Already done: {len(done)}  |  Pending: {len(pending)}")

    results_summary = {}
    failed = []
    log_lines = []
    t_start = time.perf_counter()

    for i, tf in enumerate(pending):
        tf_dir = os.path.join(args.output, tf)
        print(f"\n{'=' * 70}")
        print(f"  [{i+1}/{len(pending)}] KO: {tf}")
        print(f"{'=' * 70}")

        try:
            t0 = time.perf_counter()
            ko_tensor = virtual_knockout(wt_tensor, [tf])
            print(f"    [KO] {time.perf_counter() - t0:.1f}s")

            t0 = time.perf_counter()
            ma_result = manifold_alignment(wt_tensor, ko_tensor, d=args.ma_dim)
            print(f"    [MA] {time.perf_counter() - t0:.1f}s")

            t0 = time.perf_counter()
            d_reg = differential_regulation(ma_result, n_ko_genes=1, ko_genes=[tf])
            print(f"    [DR] {time.perf_counter() - t0:.1f}s")
            del ma_result

            os.makedirs(tf_dir, exist_ok=True)
            reg_path = os.path.join(tf_dir, f"knockout_{tf}_d_regulation.csv")
            d_reg.to_csv(reg_path, index=False)

            n_sig = (d_reg["adjusted_p-value"] < 0.05).sum()
            results_summary[tf] = {"n_sig": n_sig, "n_total": len(d_reg)}

            log_line = (
                f"{tf}\t{n_sig}\t{len(d_reg)}\t"
                f"{d_reg.iloc[0]['Gene']}\t{d_reg.iloc[0]['p-value']:.2e}\t"
                f"{'OK' if not args.enrich else 'OK+ENRICH'}"
            )
            log_lines.append(log_line)
            print(f"    Sig genes: {n_sig}/{len(d_reg)} (adj.p < 0.05)")

            if args.enrich:
                enrich_dir = os.path.join(tf_dir, "enrichment")
                run_enrichment_all(
                    dreg_df=d_reg,
                    out_dir=enrich_dir,
                    gsea_libraries=args.gsea_libs,
                    ora_libraries=args.ora_libs,
                    verbose=True,
                )
            del d_reg

        except Exception as exc:
            print(f"    FAILED: {exc}")
            traceback.print_exc()
            failed.append(tf)
            log_lines.append(f"{tf}\t0\t0\tN/A\tN/A\tFAILED:{exc}")

        if (i + 1) % 5 == 0:
            with open(os.path.join(args.output, "batch_log.tsv"), "w") as f:
                f.write("TF\tn_sig\tn_total\ttop_gene\ttop_pval\tstatus\n")
                f.write("\n".join(log_lines) + "\n")
            print(f"    [Log saved at {i+1}/{len(pending)}]")

    total_elapsed = time.perf_counter() - t_start

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  BATCH COMPLETE")
    print(f"  Finished:   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Processed:  {len(pending) - len(failed)}/{len(pending)}")
    print(f"  Total KO time: {total_elapsed/3600:.1f}h ({total_elapsed:.0f}s)")
    if failed:
        print(f"  Failed TFs: {failed}")
    print(f"{'=' * 70}")

    with open(os.path.join(args.output, "batch_log.tsv"), "w") as f:
        f.write("TF\tn_sig\tn_total\ttop_gene\ttop_pval\tstatus\n")
        f.write("\n".join(log_lines) + "\n")

    with open(os.path.join(args.output, "batch_summary.tsv"), "w") as f:
        f.write("TF\tn_sig\tn_total\n")
        for tf, info in sorted(results_summary.items()):
            f.write(f"{tf}\t{info['n_sig']}\t{info['n_total']}\n")

    print(f"\n  Log:     {os.path.join(args.output, 'batch_log.tsv')}")
    print(f"  Summary: {os.path.join(args.output, 'batch_summary.tsv')}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
