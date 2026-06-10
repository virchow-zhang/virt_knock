#!/usr/bin/env python3
"""Single-gene virtual knockout example using the virt_knock library.

Example:
  python examples/vknock_cuda.py -m expr_matrix.csv.gz -g ETS1 -o ./output
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from virt_knock import scTenifoldKnkCUDA


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-gene virtual knockout (CUDA-accelerated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-m", "--matrix", required=True,
                   help="Expression matrix (CSV/TSV, genes x cells)")
    p.add_argument("-g", "--genes", required=True,
                   help="Gene(s) to knock out (comma-separated)")
    p.add_argument("-o", "--output", default="./output",
                   help="Output directory (default: ./output)")
    p.add_argument("-n", "--n-nets", type=int, default=10,
                   help="Bootstrap networks (default: 10)")
    p.add_argument("-c", "--n-cells", type=int, default=None,
                   help="Cells per subsample (default: all cells)")
    p.add_argument("--n-comp", type=int, default=3)
    p.add_argument("--q", type=float, default=0.95,
                   help="Edge pruning quantile (default: 0.95)")
    p.add_argument("--ma-dim", type=int, default=2)
    p.add_argument("--min-lib-size", type=float, default=1000)
    p.add_argument("--min-percent", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    ko_genes = [g.strip() for g in args.genes.split(",")]

    print("=" * 60)
    print("  scTenifoldKnk — CUDA Virtual Gene Knockout")
    print("=" * 60)
    print(f"  Matrix:  {args.matrix}")
    print(f"  KO gene(s): {ko_genes}")
    print(f"  Output:  {args.output}")
    print("=" * 60)

    data = pd.read_csv(args.matrix, index_col=0)
    print(f"\n[1] Loaded: {data.shape[0]} genes x {data.shape[1]} cells")

    # Resolve KO genes (case-insensitive)
    resolved = []
    for g in ko_genes:
        if g in data.index:
            resolved.append(g)
        else:
            matches = [x for x in data.index if x.upper() == g.upper()]
            if matches:
                print(f"  '{g}' → '{matches[0]}'")
                resolved.append(matches[0])
            else:
                print(f"  WARNING: '{g}' not found in matrix, skipping")
    if not resolved:
        print("ERROR: No valid KO genes found.")
        sys.exit(1)

    sc = scTenifoldKnkCUDA(
        data=data,
        ko_genes=resolved,
        n_nets=args.n_nets,
        n_samp_cells=args.n_cells,
        n_comp=args.n_comp,
        q=args.q,
        ma_dim=args.ma_dim,
        random_state=args.seed,
        qc_kws={
            "min_lib_size": args.min_lib_size,
            "remove_outlier_cells": True,
            "min_percent": args.min_percent,
            "min_exp_avg": 0.05,
        },
    )

    t0 = time.perf_counter()
    result = sc.run()
    elapsed = time.perf_counter() - t0

    os.makedirs(args.output, exist_ok=True)
    gene_tag = "_".join(resolved[:3])
    reg_path = os.path.join(args.output, f"knockout_{gene_tag}_d_regulation.csv")
    result.to_csv(reg_path)
    sc.wt_tensor.to_csv(os.path.join(args.output, f"knockout_{gene_tag}_WT_tensor.csv"))

    print(f"\n  Results saved to: {reg_path}")
    print(f"\n  Top 20 differentially regulated genes:")
    print(result.head(20).to_string())
    print(f"\n  Total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
