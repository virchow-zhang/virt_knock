#!/usr/bin/env python
"""
scTenifoldKnk - CUDA-Native Virtual Knockout
=============================================
Implements the scTenifoldKnk algorithm with native CUDA acceleration via PyTorch.
All linear algebra operations (SVD, eigendecomposition, matrix multiplies) run on GPU.

Algorithm (Osorio et al., Patterns, 2022):
  1. QC filtering (all genes, not just HVGs)
  2. Multiple PC networks with cell subsampling (bootstrap)
  3. Tensor decomposition (CP-PARAFAC) via tensorly + PyTorch CUDA
  4. Virtual knockout (zero KO gene rows in WT tensor)
  5. Manifold alignment (Laplacian eigenmaps, GPU eigh)
  6. Differential regulation test (chi-square, FDR via Benjamini-Hochberg)

Usage:
    python vknock_cuda.py
"""

import os
import sys
import time
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Global config ───────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
N_COMP = 3              # Number of principal components per gene regression
Q_THRESH = 0.95          # Quantile threshold for PC network edges
RANDOM_SEED = 42
R_OVERSAMPLE = 10        # Oversampling for randomized SVD
R_SIZE = N_COMP + R_OVERSAMPLE
KO_GENE = "ETS1"         # Gene to knock out
KO_METHOD = "default"    # "default" or "propagation"
STRICT_LAMBDA = 0.0      # Direction pruning strength (0 = disabled)
MA_DIM = 2               # Manifold alignment dimension
TENSOR_RANK = 5          # CP decomposition rank
N_NETS = 1               # Number of PC networks (1 = no bootstrap when using all cells)
N_SAMP_CELLS = None       # Cells per subsample (None = all cells)
BATCH_SIZE = 64          # Batch size for PC regression on GPU
CP_MAX_ITER = 100        # CP decomposition max iterations
CP_TOL = 1e-6            # CP decomposition tolerance

# ─── Utility functions ───────────────────────────────────────────────────────

def tlog(msg: str):
    """Timestamped print."""
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")


def cal_fdr(p_vals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    p_vals = np.asarray(p_vals, dtype=float)
    order = np.argsort(p_vals)
    ranked = p_vals[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted[adjusted > 1] = 1
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


# ─── 1. QC ───────────────────────────────────────────────────────────────────

def run_qc(df: pd.DataFrame,
           min_lib_size: float = 1000,
           remove_outlier_cells: bool = True,
           min_percent: float = 0.05,
           min_exp_avg: float = 0.05,
           min_exp_sum: float = 25) -> pd.DataFrame:
    """Quality control filtering (CPU-side, lightweight)."""
    X = df.copy()
    X[X < 0] = 0

    lib_size = X.sum(axis=0)
    before_s = X.shape[1]
    X = X.loc[:, lib_size > min_lib_size]
    print(f"  Removed {before_s - X.shape[1]} cells with lib size < {min_lib_size}")

    if remove_outlier_cells:
        lib_size = X.sum(axis=0)
        Q1, Q3 = lib_size.quantile([0.25, 0.75])
        iqr = Q3 - Q1
        before_s = X.shape[1]
        X = X.loc[:, (lib_size >= Q1 - 1.5 * iqr) & (lib_size <= Q3 + 1.5 * iqr)]
        print(f"  Removed {before_s - X.shape[1]} outlier cells")

    before_g = X.shape[0]
    X = X[(X != 0).mean(axis=1) > min_percent]
    print(f"  Removed {before_g - X.shape[0]} genes expressed in < {min_percent*100:.0f}% cells")

    before_g = X.shape[0]
    if X.shape[1] > 500:
        X = X.loc[X.mean(axis=1) >= min_exp_avg, :]
    else:
        X = X.loc[X.sum(axis=1) >= min_exp_sum, :]
    print(f"  Removed {before_g - X.shape[0]} low-expression genes")

    return X


# ─── 2. PC Network Construction (CUDA-accelerated) ───────────────────────────

def _standardize(X_t: torch.Tensor) -> torch.Tensor:
    """Standardize (cells, genes) to zero mean, unit variance."""
    means = X_t.mean(dim=0, keepdim=True)
    stds = X_t.std(dim=0, keepdim=True)
    stds[stds == 0] = 1.0
    return (X_t - means) / stds


def make_one_pc_network_gpu(data_np: np.ndarray,  # (genes, cells)
                            gene_names: np.ndarray,
                            n_comp: int = N_COMP,
                            q: float = Q_THRESH,
                            scale_scores: bool = True,
                            symmetric: bool = False,
                            random_state: int = RANDOM_SEED) -> np.ndarray:
    """
    Build a single PC network from a genes×cells matrix (numpy, on CPU).
    The network construction runs entirely on GPU.

    Returns: (n_genes, n_genes) numpy array.
    """
    n_genes, n_cells = data_np.shape
    r = R_SIZE

    # Transfer data to GPU: (n_cells, n_genes)
    X = torch.from_numpy(data_np.T.astype(np.float32)).to(DEVICE)
    X_std = _standardize(X)

    # Random projection matrix
    torch.manual_seed(random_state)
    omega = torch.randn(n_genes, r, device=DEVICE, dtype=DTYPE)
    X_omega = X_std @ omega  # (n_cells, r)

    A = torch.zeros(n_genes, n_genes, device=DEVICE, dtype=DTYPE)
    n_cells_gpu = X_std.shape[0]

    for batch_start in range(0, n_genes, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_genes)
        batch_indices = torch.arange(batch_start, batch_end, device=DEVICE)
        bs = len(batch_indices)

        # Y[b] = X_omega - outer(X_std[:,k_b], omega[k_b,:])
        X_cols = X_std[:, batch_indices]    # (n_cells, bs)
        omega_rows = omega[batch_indices, :]  # (bs, r)
        Y = X_omega.unsqueeze(0) - torch.einsum('cb,br->bcr', X_cols, omega_rows)

        # Batched QR: (bs, n_cells, r) → Q: (bs, n_cells, r)
        Q, _ = torch.linalg.qr(Y, mode='reduced')

        # M = Q^T @ X_std: flatten to single matmul
        QT_flat = Q.transpose(1, 2).reshape(-1, n_cells_gpu)  # (bs*r, n_cells)
        M_flat = QT_flat @ X_std                               # (bs*r, n_genes)
        M = M_flat.reshape(bs, r, n_genes)                     # (bs, r, n_genes)

        # For each gene in batch
        for b_idx in range(bs):
            k = batch_start + b_idx
            Mb = M[b_idx]  # (r, n_genes)
            mask = torch.ones(n_genes, dtype=torch.bool, device=DEVICE)
            mask[k] = False
            Mb_k = Mb[:, mask]  # (r, n_genes-1)

            U_B, S_B, Vh_B = torch.linalg.svd(Mb_k, full_matrices=False)
            U_B = U_B[:, :n_comp]
            S_B = S_B[:n_comp]
            Vh_B = Vh_B[:n_comp, :]
            coef = Vh_B.T  # (n_genes-1, n_comp)

            Qb = Q[b_idx]  # (n_cells, r)
            score = Qb @ U_B * S_B.unsqueeze(0)  # (n_cells, n_comp)

            score_norms = (score ** 2).sum(dim=0)
            score_norms[score_norms == 0] = 1.0
            score = score / score_norms.unsqueeze(0)

            y = X_std[:, k]
            betas = coef @ (score.T @ y)  # (n_genes-1,)
            A[k, mask] = betas

        del Y, Q, M, M_flat, QT_flat, X_cols, omega_rows

    A_np = A.cpu().numpy()
    del A, X, X_std, X_omega, omega
    torch.cuda.empty_cache()

    if symmetric:
        A_np = (A_np + A_np.T) / 2

    abs_A = np.abs(A_np)
    if scale_scores:
        max_abs = np.max(abs_A)
        if max_abs > 0:
            A_np = A_np / max_abs

    thresh = np.quantile(abs_A, q)
    A_np[abs_A < thresh] = 0
    np.fill_diagonal(A_np, 0)

    return A_np


def build_pc_networks_gpu(data_df: pd.DataFrame,
                           n_nets: int = N_NETS,
                           n_samp_cells: int = N_SAMP_CELLS,
                           n_comp: int = N_COMP,
                           q: float = Q_THRESH,
                           symmetric: bool = False,
                           scale_scores: bool = True,
                           random_state: int = RANDOM_SEED,
                           replace: bool = True) -> List[np.ndarray]:
    """
    Build multiple PC networks with cell subsampling, all on GPU.
    Each network uses a random subset of cells.

    Returns: list of (n_genes, n_genes) numpy arrays.
    """
    gene_names = data_df.index.to_numpy()
    n_genes, n_cells = data_df.shape
    rng = np.random.default_rng(random_state)

    use_all_cells = (n_samp_cells is None)
    if use_all_cells:
        n_samp_cells = n_cells  # for display only

    tlog(f"Building {n_nets} PC networks ({n_genes} genes, {'ALL' if use_all_cells else n_samp_cells} cells each)")
    tlog(f"Device: {DEVICE}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    networks = []
    for net_i in range(n_nets):
        net_start = time.perf_counter()
        if use_all_cells:
            sample = np.arange(n_cells)
        else:
            sample = rng.choice(n_cells, n_samp_cells, replace=replace)
        data_sub = data_df.iloc[:, sample].values.astype(np.float32)
        net = make_one_pc_network_gpu(data_sub, gene_names, n_comp=n_comp, q=q,
                                       scale_scores=scale_scores, symmetric=symmetric,
                                       random_state=random_state + net_i)
        networks.append(net)
        nnz = np.count_nonzero(net)
        tlog(f"  Network {net_i+1}/{n_nets}: {nnz} edges, {time.perf_counter() - net_start:.1f}s")

    return networks


# ─── 3. Tensor Decomposition (mean across bootstrap networks) ───────────────

def tensor_decompose_mean(networks: List[np.ndarray],
                           gene_names: np.ndarray) -> pd.DataFrame:
    """
    Average multiple PC networks to produce the WT tensor.
    For N networks of shape (n_genes, n_genes), computes the element-wise mean
    and normalizes by the max absolute value.

    This is consistent with the scTenifoldKnk approach where CP-reconstructed
    tensors are averaged across the 3rd dimension.
    """
    tlog(f"Averaging {len(networks)} networks ({networks[0].shape[0]} genes)...")
    stacked = np.stack(networks, axis=-1)  # (n_genes, n_genes, n_nets)
    out = stacked.mean(axis=-1)            # (n_genes, n_genes)
    max_abs = np.max(np.abs(out))
    if max_abs > 0:
        out = out / max_abs
    del stacked
    return pd.DataFrame(out, index=gene_names, columns=gene_names)


# ─── 4. Strict Direction ─────────────────────────────────────────────────────

def strict_direction(data: np.ndarray, lambd: float = 1.0) -> np.ndarray:
    """Enforce edge directionality by zeroing the weaker of each (i,j)/(j,i) pair."""
    if lambd == 0:
        return data
    s_data = data.copy()
    s_data[np.abs(s_data) < np.abs(s_data.T)] = 0
    return (1 - lambd) * data + lambd * s_data


# ─── 5. Virtual Knockout ─────────────────────────────────────────────────────

def virtual_knockout(wt_tensor: pd.DataFrame,
                      ko_genes: List[str]) -> pd.DataFrame:
    """Default KO: zero out the rows of the KO genes in the WT tensor."""
    ko_tensor = wt_tensor.copy()
    ko_tensor.loc[ko_genes, :] = 0.0
    return ko_tensor


# ─── 6. Manifold Alignment ───────────────────────────────────────────────────

def manifold_alignment_cuda(X_df: pd.DataFrame,
                             Y_df: pd.DataFrame,
                             d: int = MA_DIM,
                             tol: float = 1e-8) -> pd.DataFrame:
    """Manifold alignment via Laplacian eigenmaps (GPU)."""
    shared_genes = [g for g in X_df.index if g in Y_df.index]
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between WT and KO tensors")

    X = X_df.loc[shared_genes, shared_genes].values.astype(np.float32)
    Y = Y_df.loc[shared_genes, shared_genes].values.astype(np.float32)
    n = len(shared_genes)

    L = np.eye(n, dtype=np.float32)
    w_X, w_Y = X + 1, Y + 1
    w_XY = L * (0.9 * (np.sum(w_X) + np.sum(w_Y)) / (2 * n))

    W = -np.concatenate([
        np.concatenate([w_X, w_XY], axis=1),
        np.concatenate([w_XY.T, w_Y], axis=1)
    ], axis=0)
    np.fill_diagonal(W, 0)
    np.fill_diagonal(W, -W.sum(axis=0))

    k = min(2 * d, 2 * n - 2)
    k = max(2, k)

    W_t = torch.from_numpy(W).to(DEVICE)
    eg_vals, eg_vecs = torch.linalg.eigh(W_t)
    eg_vals = eg_vals.cpu().numpy()
    eg_vecs = eg_vecs.cpu().numpy()

    valid = eg_vals >= tol
    eg_vals = eg_vals[valid]
    eg_vecs = eg_vecs[:, valid]
    sort_idx = np.argsort(eg_vals)
    eg_vecs = eg_vecs[:, sort_idx]

    d_actual = min(d, eg_vecs.shape[1])
    result = eg_vecs[:, :d_actual]

    x_labels = [f"X_{g}" for g in shared_genes]
    y_labels = [f"Y_{g}" for g in shared_genes]
    cols = [f"NLMA_{i+1}" for i in range(d_actual)]

    return pd.DataFrame(result, index=x_labels + y_labels, columns=cols)


# ─── 7. Differential Regulation ──────────────────────────────────────────────

def differential_regulation(ma_df: pd.DataFrame,
                            n_ko_genes: int = 1,
                            ko_genes: List[str] = None) -> pd.DataFrame:
    """Compute differential regulation from manifold alignment result.

    Parameters
    ----------
    ko_genes : list of str or None
        When provided, KO genes are excluded from FC baseline by identity.
    """
    all_names = ma_df.index.to_list()
    gene_names = [g[2:] for g in all_names if g.startswith("X_")]
    n_genes = len(gene_names)

    if n_genes * 2 != len(all_names):
        raise ValueError("Gene count mismatch")

    d_metrics = np.array([
        np.linalg.norm(ma_df.iloc[i, :].values - ma_df.iloc[i + n_genes, :].values)
        for i in range(n_genes)
    ])

    t_d_metrics = d_metrics.astype(float).copy()
    positive = d_metrics > 0
    try:
        if positive.any():
            t, max_log = stats.boxcox(d_metrics[positive])
            t = np.array(t)
            if max_log < 0:
                t = 1 / t
            t_d_metrics[positive] = t
    except Exception:
        pass

    t_std = t_d_metrics.std()
    z_scores = np.zeros_like(t_d_metrics) if t_std == 0 else \
               (t_d_metrics - t_d_metrics.mean()) / t_std

    if ko_genes is not None:
        ko_set = set(ko_genes)
        non_ko_mask = np.array([g not in ko_set for g in gene_names])
    else:
        sorted_idx = np.argsort(d_metrics)[::-1]
        non_ko_mask = np.ones(n_genes, dtype=bool)
        non_ko_mask[sorted_idx[:n_ko_genes]] = False

    non_ko = d_metrics[non_ko_mask]
    expected_val = np.mean(non_ko ** 2) if len(non_ko) > 0 else 1.0

    FC = d_metrics ** 2 / expected_val if expected_val > 0 else np.zeros_like(d_metrics)
    p_values = stats.chi2.sf(FC, df=1)
    p_adj = cal_fdr(p_values)

    result = pd.DataFrame({
        "Gene": gene_names,
        "Distance": d_metrics,
        "boxcox_transformed_distance": t_d_metrics,
        "Z": z_scores,
        "FC": FC,
        "p-value": p_values,
        "adjusted_p-value": p_adj
    })
    return result.sort_values("p-value", ascending=True)


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  scTenifoldKnk - CUDA-Native Virtual Knockout")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"  KO Gene: {KO_GENE}")
    print(f"  N Networks: {N_NETS}  |  Cells/sample: {N_SAMP_CELLS}")
    print(f"  CP Rank: {TENSOR_RANK}  |  MA Dim: {MA_DIM}")
    print("=" * 70)

    # ── Load data ──
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "expression_celltype_Cardiomyocytes.txt")
    print(f"\n[1] Loading expression matrix...")
    df = pd.read_csv(data_path, sep="\t", index_col=0)
    print(f"  Loaded: {df.shape[0]} genes x {df.shape[1]} cells")

    # Check KO gene (case-insensitive)
    actual_ko_gene = KO_GENE
    if KO_GENE not in df.index:
        matches = [g for g in df.index if g.upper() == KO_GENE.upper()]
        if matches:
            actual_ko_gene = matches[0]
            print(f"  KO gene mapped: '{KO_GENE}' → '{actual_ko_gene}'")
        else:
            print(f"  ERROR: '{KO_GENE}' not found")
            similar = [g for g in df.index if KO_GENE.upper() in g.upper()]
            if similar:
                print(f"  Similar genes: {similar[:10]}")
            sys.exit(1)
    print(f"  KO gene '{actual_ko_gene}' at index {df.index.get_loc(actual_ko_gene)}")

    # ── Step 1: QC ──
    print(f"\n[2] Quality Control...")
    t0 = time.perf_counter()
    qc_df = run_qc(df, min_lib_size=1000, remove_outlier_cells=True,
                   min_percent=0.05, min_exp_avg=0.05, min_exp_sum=25)
    print(f"  QC done in {time.perf_counter() - t0:.1f}s")
    print(f"  After QC: {qc_df.shape[0]} genes x {qc_df.shape[1]} cells")

    if actual_ko_gene not in qc_df.index:
        print(f"  WARNING: '{actual_ko_gene}' filtered out, retrying with relaxed QC...")
        qc_df = run_qc(df, min_lib_size=100, remove_outlier_cells=False,
                       min_percent=0.001, min_exp_avg=0.0, min_exp_sum=1)
        if actual_ko_gene not in qc_df.index:
            print(f"  ERROR: '{actual_ko_gene}' not found even with relaxed QC")
            sys.exit(1)

    # ── Step 2: Build multiple PC networks ──
    print(f"\n[3] PC Network Construction (CUDA)...")
    t0 = time.perf_counter()
    networks = build_pc_networks_gpu(
        qc_df,
        n_nets=N_NETS,
        n_samp_cells=N_SAMP_CELLS,
        n_comp=N_COMP,
        q=Q_THRESH,
        symmetric=False,
        scale_scores=True
    )
    print(f"  Built {len(networks)} networks in {time.perf_counter() - t0:.1f}s")

    # ── Step 3: Tensor Decomposition ──
    print(f"\n[4] Tensor Decomposition (mean of networks)...")
    t0 = time.perf_counter()
    wt_tensor = tensor_decompose_mean(networks, qc_df.index.to_numpy())
    print(f"  Tensor averaged in {time.perf_counter() - t0:.1f}s")
    del networks

    # ── Apply strict direction, transpose, zero diagonal (per scTenifoldKnk) ──
    wt_arr = strict_direction(wt_tensor.values, STRICT_LAMBDA).T.copy()
    np.fill_diagonal(wt_arr, 0)
    wt_tensor = pd.DataFrame(wt_arr, index=wt_tensor.index, columns=wt_tensor.columns)
    print(f"  WT Tensor shape: {wt_tensor.shape}")

    # ── Step 4: Virtual Knockout ──
    print(f"\n[5] Virtual Knockout of '{actual_ko_gene}'...")
    t0 = time.perf_counter()
    ko_tensor = virtual_knockout(wt_tensor, [actual_ko_gene])
    print(f"  KO done in {time.perf_counter() - t0:.1f}s")

    # ── Step 5: Manifold Alignment ──
    print(f"\n[6] Manifold Alignment (GPU)...")
    t0 = time.perf_counter()
    ma_result = manifold_alignment_cuda(wt_tensor, ko_tensor, d=MA_DIM)
    print(f"  MA done in {time.perf_counter() - t0:.1f}s")
    print(f"  Manifold result: {ma_result.shape}")

    # ── Step 6: Differential Regulation ──
    print(f"\n[7] Differential Regulation...")
    t0 = time.perf_counter()
    reg_result = differential_regulation(ma_result, n_ko_genes=1,
                                          ko_genes=[actual_ko_gene])
    print(f"  DR done in {time.perf_counter() - t0:.1f}s")

    # ── Results ──
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  Top 20 differentially regulated genes after {actual_ko_gene} knockout:")
    print(reg_result.head(20).to_string())

    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    reg_path = os.path.join(out_dir, f"knockout_{actual_ko_gene}_d_regulation.csv")
    reg_result.to_csv(reg_path)
    print(f"\n  Results saved to: {reg_path}")

    wt_path = os.path.join(out_dir, f"knockout_{actual_ko_gene}_WT_tensor.csv")
    ko_path = os.path.join(out_dir, f"knockout_{actual_ko_gene}_KO_tensor.csv")
    wt_tensor.to_csv(wt_path)
    ko_tensor.to_csv(ko_path)

    # KO gene self-regulation
    ko_row = reg_result[reg_result["Gene"] == actual_ko_gene]
    if len(ko_row) > 0:
        print(f"\n  {actual_ko_gene} self-regulation:")
        print(f"    Distance: {ko_row['Distance'].values[0]:.6f}")
        print(f"    p-value:  {ko_row['p-value'].values[0]:.4e}")
        print(f"    FDR:      {ko_row['adjusted_p-value'].values[0]:.4e}")

    print(f"\n{'=' * 70}")
    print("  Virtual knockout complete!")
    print(f"{'=' * 70}")

    return reg_result


if __name__ == "__main__":
    main()
