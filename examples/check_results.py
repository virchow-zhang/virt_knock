#!/usr/bin/env python3
"""Summarize batch TF knockout results.

Example:
  python examples/check_results.py --output ./output_tf_batch
"""

import argparse
import os
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Summarize batch TF knockout results")
    p.add_argument("-o", "--output", required=True,
                   help="Batch output directory containing batch_summary.tsv")
    p.add_argument("-n", "--top", type=int, default=15,
                   help="Number of top TFs to show (default: 15)")
    return p.parse_args()


def main():
    args = parse_args()
    summary_path = os.path.join(args.output, "batch_summary.tsv")

    if not os.path.exists(summary_path):
        print(f"ERROR: {summary_path} not found. Has the batch pipeline completed?")
        return

    s = pd.read_csv(summary_path, sep="\t")
    if "TF" not in s.columns or "n_sig" not in s.columns:
        print(f"ERROR: unexpected format in {summary_path}")
        return

    total_sig = s["n_sig"].sum()
    n_total = s["n_total"].iloc[0] if "n_total" in s.columns else "?"
    mean_sig = s["n_sig"].mean()

    print(f"=== BATCH TF KNOCKOUT RESULTS ===")
    print(f"Total genes tested: {n_total}")
    print(f"TF knockout genes:  {len(s)}")
    print(f"Total significant genes (adj.p < 0.05): {total_sig}")
    print(f"Mean sig per TF: {mean_sig:.1f}")
    print()

    top = s.nlargest(args.top, "n_sig")
    print(f"TOP {args.top} TFs by perturbation magnitude:")
    for _, r in top.iterrows():
        n_t = r.get("n_total", n_total)
        pct = r["n_sig"] / n_t * 100 if n_t else 0
        print(f"  {r['TF']:15s}  {r['n_sig']:4d} / {n_t:4d}  ({pct:.2f}%)")

    print()
    bins = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 9999)]
    for lo, hi in bins:
        n = ((s["n_sig"] > lo) & (s["n_sig"] <= hi)).sum()
        if n > 0:
            lo_str = str(lo + 1) if lo == 0 else str(lo + 1)
            print(f"  {lo_str}-{hi} sig genes: {n} TFs")


if __name__ == "__main__":
    main()
