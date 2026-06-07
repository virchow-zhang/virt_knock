"""
Enrichment analysis for virtual gene knockout results.

Provides GSEA prerank (all genes ranked by Z-score) and ORA (Enrichr)
against GO Biological Process, KEGG, and Reactome gene set libraries.
"""
import os
import sys
import time
from typing import List, Optional

import pandas as pd

try:
    import gseapy as gp
    _HAS_GSEAPY = True
except ImportError:
    _HAS_GSEAPY = False


# Default gene set libraries
DEFAULT_GSEA_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2019_Mouse",
    "Reactome_2022",
]

DEFAULT_ORA_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2019_Mouse",
    "Reactome_2022",
]

GSEA_MIN_SIZE = 10
GSEA_MAX_SIZE = 500
N_PERM = 1000
SEED = 42


def _check_gseapy():
    if not _HAS_GSEAPY:
        raise ImportError(
            "gseapy is required for enrichment analysis. "
            "Install it with: pip install gseapy"
        )


def run_gsea_prerank(
    dreg_df: pd.DataFrame,
    out_dir: str,
    libraries: Optional[List[str]] = None,
    min_size: int = GSEA_MIN_SIZE,
    max_size: int = GSEA_MAX_SIZE,
    n_perm: int = N_PERM,
    seed: int = SEED,
    verbose: bool = True,
) -> dict:
    """Run GSEA prerank analysis on differential regulation results.

    All genes are ranked by Z-score from the virtual knockout output.

    Parameters
    ----------
    dreg_df : pd.DataFrame
        Output of ``differential_regulation`` with columns: Gene, Z.
    out_dir : str
        Output directory for GSEA results.
    libraries : list of str, optional
        Enrichr gene set library names.  Defaults to GO BP, KEGG Mouse, Reactome.
    min_size, max_size : int
        Gene set size filters.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Library name → prerank result object.
    """
    _check_gseapy()

    if libraries is None:
        libraries = DEFAULT_GSEA_LIBRARIES

    os.makedirs(out_dir, exist_ok=True)

    # Build preranked list: gene → Z-score
    prerank = dreg_df.set_index("Gene")["Z"].dropna().sort_values(ascending=False)
    prerank = prerank[~prerank.index.duplicated(keep="first")]

    if verbose:
        print(f"\n  [GSEA] Preranked {len(prerank)} genes "
              f"(Z range: {prerank.max():.2f} to {prerank.min():.2f})")

    results = {}
    for lib in libraries:
        lib_short = lib.replace(" ", "_")
        lib_dir = os.path.join(out_dir, f"gsea_{lib_short}")
        if verbose:
            print(f"  Running GSEA prerank: {lib} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        try:
            pre_res = gp.prerank(
                rnk=prerank,
                gene_sets=lib,
                outdir=lib_dir,
                min_size=min_size,
                max_size=max_size,
                permutation_num=n_perm,
                seed=seed,
                no_plot=False,
                verbose=False,
            )
            results[lib] = pre_res
            n_sig = (pre_res.res2d['FDR q-val'] < 0.25).sum()
            if verbose:
                print(f"OK ({n_sig}/{len(pre_res.res2d)} sig, "
                      f"{time.perf_counter()-t0:.1f}s)")
        except Exception as exc:
            if verbose:
                print(f"FAILED: {exc}")

    return results


def run_ora_enrichr(
    gene_list: List[str],
    out_dir: str,
    libraries: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """Run over-representation analysis (ORA) via Enrichr.

    Parameters
    ----------
    gene_list : list of str
        Gene symbols to test for enrichment.
    out_dir : str
        Output directory for ORA results.
    libraries : list of str, optional
        Enrichr gene set library names.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Library name → enrichr result object.
    """
    _check_gseapy()

    if libraries is None:
        libraries = DEFAULT_ORA_LIBRARIES

    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"\n  [ORA] Testing {len(gene_list)} significant genes")

    results = {}
    for lib in libraries:
        lib_short = lib.replace(" ", "_")
        lib_dir = os.path.join(out_dir, f"ora_{lib_short}")
        if verbose:
            print(f"  Running ORA: {lib} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=lib,
                outdir=lib_dir,
                no_plot=False,
                verbose=False,
            )
            results[lib] = enr
            n_sig = (enr.results['Adjusted P-value'] < 0.05).sum()
            if verbose:
                print(f"OK ({n_sig}/{len(enr.results)} sig, "
                      f"{time.perf_counter()-t0:.1f}s)")
        except Exception as exc:
            if verbose:
                print(f"FAILED: {exc}")

    return results


def run_enrichment_all(
    dreg_df: pd.DataFrame,
    out_dir: str,
    ora_p_cutoff: float = 0.05,
    gsea_libraries: Optional[List[str]] = None,
    ora_libraries: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """Run full enrichment suite (GSEA + ORA) on knockout results.

    Parameters
    ----------
    dreg_df : pd.DataFrame
        Output of ``differential_regulation``.
    out_dir : str
        Base output directory.  GSEA/ORA results go into subdirectories.
    ora_p_cutoff : float
        Adjusted p-value threshold for selecting ORA genes.
    gsea_libraries, ora_libraries : list of str, optional
        Gene set libraries.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        {"gsea": {...}, "ora": {...}}
    """
    _check_gseapy()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Enrichment Analysis")
        print(f"{'='*60}")

    # Determine adjusted p-value column
    adjp_col = "adjusted_p-value"
    if adjp_col not in dreg_df.columns and "adjusted_p-value" not in dreg_df.columns:
        adjp_col = "adjusted_p-value"
    if adjp_col not in dreg_df.columns:
        # Try common variants
        for col in dreg_df.columns:
            if "adjusted" in col.lower() or "adj" in col.lower():
                adjp_col = col
                break

    results = {}

    # 1. GSEA prerank
    results["gsea"] = run_gsea_prerank(
        dreg_df=dreg_df,
        out_dir=os.path.join(out_dir, "gsea"),
        libraries=gsea_libraries,
        verbose=verbose,
    )

    # 2. ORA
    sig_mask = dreg_df[adjp_col] < ora_p_cutoff
    sig_genes = dreg_df.loc[sig_mask, "Gene"].dropna().tolist()

    if len(sig_genes) < 3:
        # Relax to nominal p < 0.05
        p_col = "p-value"
        if p_col not in dreg_df.columns:
            for col in dreg_df.columns:
                if "p-value" in col.lower() or col == "p-value":
                    p_col = col
                    break
        sig_genes = dreg_df[dreg_df[p_col] < 0.05]["Gene"].dropna().tolist()
        if verbose:
            print(f"  (Relaxed ORA cutoff: {len(sig_genes)} genes, p<0.05)")

    if len(sig_genes) >= 3:
        results["ora"] = run_ora_enrichr(
            gene_list=sig_genes,
            out_dir=os.path.join(out_dir, "ora"),
            libraries=ora_libraries,
            verbose=verbose,
        )
    elif verbose:
        print("  [ORA] Skipped -- too few significant genes.")

    if verbose:
        print(f"\n  Enrichment complete.")
        _print_top_enrichment(out_dir)

    return results


def _print_top_enrichment(out_dir: str, top_n: int = 3):
    """Print top enrichment hits from each analysis."""
    # GSEA
    for lib in DEFAULT_GSEA_LIBRARIES:
        lib_short = lib.replace(" ", "_")
        report = os.path.join(out_dir, "gsea", f"gsea_{lib_short}",
                              "gseapy.gene_set.prerank.report.csv")
        if os.path.exists(report):
            df = pd.read_csv(report, index_col=0)
            top = df[df['FDR q-val'] < 0.25].head(top_n)
            if len(top) > 0:
                print(f"\n  Top GSEA -- {lib}:")
                for _, row in top.iterrows():
                    direction = "^" if row['NES'] > 0 else "v"
                    print(f"    {direction} {row['Term'][:60]} "
                          f"(NES={row['NES']:+.2f}, FDR={row['FDR q-val']:.2e})")

    # ORA
    for lib in DEFAULT_ORA_LIBRARIES:
        lib_short = lib.replace(" ", "_")
        for prefix in ["Enrichr.human.enrichr.reports.txt",
                       "Enrichr.mouse.enrichr.reports.txt"]:
            report = os.path.join(out_dir, "ora", f"ora_{lib_short}", prefix)
            if os.path.exists(report):
                df = pd.read_csv(report, sep="\t")
                top = df[df["Adjusted P-value"] < 0.05].head(top_n)
                if len(top) > 0:
                    print(f"\n  Top ORA -- {lib}:")
                    for _, row in top.iterrows():
                        print(f"    * {row['Term'][:60]} "
                              f"(overlap={row['Overlap']}, adj.p={row['Adjusted P-value']:.2e})")
                break
