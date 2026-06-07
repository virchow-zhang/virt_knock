"""Command-line interface for virt_knock."""

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd

from .core import scTenifoldKnkCUDA


def main():
    parser = argparse.ArgumentParser(
        description="virt_knock — CUDA-native single-cell virtual gene knockout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  virt_knock -i expr.tsv -g ETS1
  virt_knock -i expr.tsv -g ETS1 -n 10 -c 500
  virt_knock -i expr.tsv -g ETS1,FOXP3 --all-cells --no-bootstrap
""",
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to expression matrix (tsv/csv, genes × cells)")
    parser.add_argument("-g", "--genes", required=True,
                        help="Gene name(s) to knock out (comma-separated)")
    parser.add_argument("-o", "--output", default="./output",
                        help="Output directory (default: ./output_TAG_TIMESTAMP)")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Do not append timestamp to output directory")
    parser.add_argument("--enrich", action="store_true",
                        help="Run enrichment analysis (GSEA + ORA) after knockout")
    parser.add_argument("-n", "--n-nets", type=int, default=10,
                        help="Number of bootstrap networks (default: 10)")
    parser.add_argument("-c", "--n-cells", type=int, default=500,
                        help="Cells per subsample (default: 500)")
    parser.add_argument("--all-cells", action="store_true",
                        help="Use all cells (no subsampling)")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Use all cells in each network (no cell subsampling)")
    parser.add_argument("--n-comp", type=int, default=3,
                        help="PC components per gene regression (default: 3)")
    parser.add_argument("--q", type=float, default=0.95,
                        help="Quantile threshold for edge pruning (default: 0.95)")
    parser.add_argument("--strict-lambda", type=float, default=0.0,
                        help="Direction pruning strength (default: 0)")
    parser.add_argument("--ma-dim", type=int, default=2,
                        help="Manifold alignment dimension (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--min-lib-size", type=float, default=1000,
                        help="Min library size per cell (default: 1000)")
    parser.add_argument("--min-percent", type=float, default=0.05,
                        help="Min fraction of cells expressing a gene (default: 0.05)")
    parser.add_argument("--sep", default="\t",
                        help="Field separator in input file (default: tab)")

    args = parser.parse_args()

    # Resolve arguments
    n_samp_cells = None if (args.all_cells or args.no_bootstrap) else args.n_cells
    n_nets = args.n_nets
    ko_genes = [g.strip() for g in args.genes.split(",")]

    # Defaults warning
    if not args.all_cells and not args.no_bootstrap and n_samp_cells <= 500:
        print(f"  [!] Subsampling {n_samp_cells} cells may miss KO gene effects.")
        print(f"      Consider --all-cells or -c with a larger value for robust results.")

    # Timestamped output directory
    if not args.no_timestamp:
        tag = "_".join(ko_genes[:3])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(args.output, f"{tag}_n{n_nets}_{ts}")

    print("=" * 60)
    print("  virt_knock — CUDA Virtual Gene Knockout")
    print("=" * 60)
    print(f"  Input:      {args.input}")
    print(f"  KO genes:   {ko_genes}")
    print(f"  N networks: {n_nets}  |  Cells/sample: {'ALL' if n_samp_cells is None else n_samp_cells}")
    print(f"  Output:     {args.output}")
    if args.enrich:
        print(f"  Enrichment: GSEA + ORA (GO, KEGG, Reactome)")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data ...")
    sep = "\t" if args.sep.lower() in ("tab", "\\t", "\t") else args.sep
    data = pd.read_csv(args.input, sep=sep, index_col=0)
    print(f"  Loaded: {data.shape[0]} genes × {data.shape[1]} cells")

    # Check KO genes (case-insensitive)
    resolved = []
    for g in ko_genes:
        if g in data.index:
            resolved.append(g)
        else:
            matches = [x for x in data.index if x.upper() == g.upper()]
            if matches:
                print(f"  Note: '{g}' → '{matches[0]}'")
                resolved.append(matches[0])
            else:
                print(f"  Warning: '{g}' not found, skipping")
    if not resolved:
        print("  ERROR: no valid KO genes found")
        sys.exit(1)

    # Run pipeline
    sc = scTenifoldKnkCUDA(
        data=data,
        ko_genes=resolved,
        n_nets=n_nets,
        n_samp_cells=n_samp_cells,
        n_comp=args.n_comp,
        q=args.q,
        strict_lambda=args.strict_lambda,
        ma_dim=args.ma_dim,
        random_state=args.seed,
        qc_kws={
            "min_lib_size": args.min_lib_size,
            "remove_outlier_cells": True,
            "min_percent": args.min_percent,
            "min_exp_avg": 0.05,
            "min_exp_sum": 25,
        },
    )

    t0 = time.perf_counter()
    result = sc.run()
    elapsed = time.perf_counter() - t0

    # Save
    os.makedirs(args.output, exist_ok=True)
    gene_tag = "_".join(resolved[:3])
    reg_path = os.path.join(args.output, f"knockout_{gene_tag}_d_regulation.csv")
    result.to_csv(reg_path)
    print(f"\n  Results saved to: {reg_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Top 20 differentially regulated genes:")
    print(f"{'=' * 60}")
    print(result.head(20).to_string())
    print(f"\n  Total wall time: {elapsed:.1f}s")

    # Enrichment (optional)
    if args.enrich:
        from .enrichment import run_enrichment_all
        run_enrichment_all(
            dreg_df=result,
            out_dir=os.path.join(args.output, "enrichment"),
            verbose=True,
        )
