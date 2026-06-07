"""
Core algorithm: scTenifoldKnk pipeline with CUDA acceleration.

Reference
---------
Osorio, D., Zhong, Y., Li, G., Xu, Q., Yang, Y., Tian, Y., Chapkin, R. S.,
Huang, J. Z., & Cai, J. J. (2022). scTenifoldKnk: An efficient virtual
knockout tool for gene function predictions via single-cell gene regulatory
network perturbation. Patterns, 3(3), 100434.
"""

import time
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Global defaults ─────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# PC network parameters
N_COMP = 3              # Principal components per gene regression
Q_THRESH = 0.95          # Quantile threshold for edge pruning
R_OVERSAMPLE = 10        # Oversampling for randomized SVD
R_SIZE = N_COMP + R_OVERSAMPLE
BATCH_SIZE = 64          # Genes per GPU batch

# Pipeline parameters
KO_METHOD = "default"    # "default" or "propagation"
STRICT_LAMBDA = 0.0
MA_DIM = 2

DEFAULT_RANDOM_SEED = 42

# ─── Utilities ───────────────────────────────────────────────────────────────

def _tlog(msg: str):
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


# ─── 1. Quality Control ─────────────────────────────────────────────────────

def run_qc(
    df: pd.DataFrame,
    min_lib_size: float = 1000,
    remove_outlier_cells: bool = True,
    min_percent: float = 0.05,
    min_exp_avg: float = 0.05,
    min_exp_sum: float = 25,
) -> pd.DataFrame:
    """Filter genes and cells by standard single-cell QC metrics.

    Parameters
    ----------
    df : pd.DataFrame
        genes × cells expression matrix.
    min_lib_size : float
        Minimum library size per cell.
    remove_outlier_cells : bool
        Remove cells whose library size falls outside 1.5 × IQR.
    min_percent : float
        Keep genes expressed in at least this fraction of cells.
    min_exp_avg : float
        Minimum mean expression (used when n_cells > 500).
    min_exp_sum : float
        Minimum sum expression (used when n_cells <= 500).

    Returns
    -------
    pd.DataFrame
        Filtered genes × cells matrix.
    """
    X = df.copy()
    X[X < 0] = 0

    lib_size = X.sum(axis=0)
    before = X.shape[1]
    X = X.loc[:, lib_size > min_lib_size]
    print(f"  Cells removed (lib size < {min_lib_size}): {before - X.shape[1]}")

    if remove_outlier_cells:
        lib_size = X.sum(axis=0)
        q1, q3 = lib_size.quantile([0.25, 0.75])
        iqr = q3 - q1
        before = X.shape[1]
        mask = (lib_size >= q1 - 1.5 * iqr) & (lib_size <= q3 + 1.5 * iqr)
        X = X.loc[:, mask]
        print(f"  Outlier cells removed: {before - X.shape[1]}")

    before = X.shape[0]
    X = X[(X != 0).mean(axis=1) > min_percent]
    print(f"  Genes removed (< {min_percent*100:.0f}% cells): {before - X.shape[0]}")

    before = X.shape[0]
    if X.shape[1] > 500:
        X = X.loc[X.mean(axis=1) >= min_exp_avg, :]
    else:
        X = X.loc[X.sum(axis=1) >= min_exp_sum, :]
    print(f"  Low-expression genes removed: {before - X.shape[0]}")

    return X


# ─── 2. PC Network Construction (CUDA) ──────────────────────────────────────

def _standardize(X_t: torch.Tensor) -> torch.Tensor:
    """(cells, genes) -> zero mean, unit variance."""
    means = X_t.mean(dim=0, keepdim=True)
    stds = X_t.std(dim=0, keepdim=True)
    stds[stds == 0] = 1.0
    return (X_t - means) / stds


def _make_one_pc_network(
    data_np: np.ndarray,
    gene_names: np.ndarray,
    n_comp: int = N_COMP,
    q: float = Q_THRESH,
    scale_scores: bool = True,
    symmetric: bool = False,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """Build a single PC network on GPU via batched randomized-SVD regression.

    For each gene k, regress y = X[:,k] on Xi = X[:, ~k] using a low-rank
    (n_comp) approximation obtained from randomized SVD.

    Optimisation: precompute XΩ once, then for each gene k the random
    projection of Xi is obtained via the rank-1 update
        Xi_omega_k = XΩ - outer(x_k, ω_row_k).
    """
    n_genes, n_cells = data_np.shape
    r = R_SIZE

    X = torch.from_numpy(data_np.T.astype(np.float32)).to(DEVICE)
    X_std = _standardize(X)

    torch.manual_seed(random_state)
    omega = torch.randn(n_genes, r, device=DEVICE, dtype=DTYPE)
    X_omega = X_std @ omega

    A = torch.zeros(n_genes, n_genes, device=DEVICE, dtype=DTYPE)

    for batch_start in range(0, n_genes, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_genes)
        batch_idx = torch.arange(batch_start, batch_end, device=DEVICE)
        bs = len(batch_idx)

        X_cols = X_std[:, batch_idx]
        omega_rows = omega[batch_idx, :]
        Y = X_omega.unsqueeze(0) - torch.einsum("cb,br->bcr", X_cols, omega_rows)

        Q, _ = torch.linalg.qr(Y, mode="reduced")              # (bs, ncells, r)
        QT_f = Q.transpose(1, 2).reshape(-1, X_std.shape[0])   # (bs*r, ncells)
        M_f = QT_f @ X_std                                     # (bs*r, n_genes)
        M = M_f.reshape(bs, r, n_genes)                        # (bs, r, n_genes)

        for b in range(bs):
            k = batch_start + b
            mask = torch.ones(n_genes, dtype=torch.bool, device=DEVICE)
            mask[k] = False
            Mb_k = M[b][:, mask]                                # (r, n_genes-1)

            U_B, S_B, Vh_B = torch.linalg.svd(Mb_k, full_matrices=False)
            U_B, S_B, Vh_B = U_B[:, :n_comp], S_B[:n_comp], Vh_B[:n_comp, :]
            coef = Vh_B.T                                       # (n_genes-1, n_comp)

            score = Q[b] @ U_B * S_B.unsqueeze(0)               # (ncells, n_comp)
            nrm = (score ** 2).sum(dim=0)
            nrm[nrm == 0] = 1.0
            score = score / nrm.unsqueeze(0)

            y = X_std[:, k]
            betas = coef @ (score.T @ y)
            A[k, mask] = betas

        del Y, Q, M, M_f, QT_f, X_cols, omega_rows

    A_np = A.cpu().numpy()
    del A, X, X_std, X_omega, omega
    torch.cuda.empty_cache()

    if symmetric:
        A_np = (A_np + A_np.T) / 2

    abs_A = np.abs(A_np)
    if scale_scores:
        vmax = np.max(abs_A)
        if vmax > 0:
            A_np = A_np / vmax

    A_np[abs_A < np.quantile(abs_A, q)] = 0
    np.fill_diagonal(A_np, 0)

    return A_np


def build_pc_networks(
    data_df: pd.DataFrame,
    n_nets: int = 10,
    n_samp_cells: Optional[int] = 500,
    n_comp: int = N_COMP,
    q: float = Q_THRESH,
    symmetric: bool = False,
    scale_scores: bool = True,
    random_state: int = DEFAULT_RANDOM_SEED,
    replace: bool = True,
    verbose: bool = True,
) -> List[np.ndarray]:
    """Build multiple PC networks with cell subsampling (bootstrap).

    Each network is built on a random subset of cells.  Set
    ``n_samp_cells=None`` to use all cells (single network recommended).

    Parameters
    ----------
    data_df : pd.DataFrame
        genes × cells expression matrix (post-QC).
    n_nets : int
        Number of bootstrap networks.
    n_samp_cells : int or None
        Cells per subsample.  None = all cells.
    n_comp : int
        PC components per gene regression.
    q : float
        Quantile threshold for edge pruning.
    symmetric : bool
        Force symmetric adjacency.
    scale_scores : bool
        Normalise by max absolute weight.
    random_state : int
        Seed for reproducibility.
    replace : bool
        Sample cells with replacement.
    verbose : bool
        Print progress.

    Returns
    -------
    list of np.ndarray
        Each element is a (n_genes, n_genes) adjacency matrix.
    """
    gene_names = data_df.index.to_numpy()
    n_genes, n_cells = data_df.shape
    rng = np.random.default_rng(random_state)
    use_all = n_samp_cells is None

    if verbose:
        label = "ALL" if use_all else str(n_samp_cells)
        _tlog(f"Building {n_nets} PC networks ({n_genes} genes, {label} cells)")
        if DEVICE.type == "cuda":
            p = torch.cuda.get_device_properties(0)
            print(f"  Device: {DEVICE}  |  {p.name}  |  {p.total_memory/1e9:.1f} GB")

    networks = []
    for net_i in range(n_nets):
        t0 = time.perf_counter()
        if use_all:
            sample = np.arange(n_cells)
        else:
            sample = rng.choice(n_cells, n_samp_cells, replace=replace)

        data_sub = data_df.iloc[:, sample].values.astype(np.float32)
        net = _make_one_pc_network(
            data_sub, gene_names,
            n_comp=n_comp, q=q,
            scale_scores=scale_scores, symmetric=symmetric,
            random_state=random_state + net_i,
        )
        networks.append(net)
        if verbose:
            nnz = np.count_nonzero(net)
            _tlog(f"  Net {net_i+1}/{n_nets}: {nnz} edges, "
                  f"{time.perf_counter()-t0:.1f}s")

    return networks


# ─── 3. Tensor Decomposition ─────────────────────────────────────────────────

def tensor_decompose_mean(
    networks: List[np.ndarray],
    gene_names: np.ndarray,
    verbose: bool = True,
) -> pd.DataFrame:
    """Average bootstrap networks into a single WT tensor.

    Equivalent to the scTenifoldKnk CP-reconstruction averaged across
    the 3rd dimension, simplified to the element-wise mean for efficiency.
    """
    if verbose:
        _tlog(f"Averaging {len(networks)} networks → WT tensor")

    stacked = np.stack(networks, axis=-1)
    out = stacked.mean(axis=-1)
    vmax = np.max(np.abs(out))
    if vmax > 0:
        out = out / vmax
    del stacked
    return pd.DataFrame(out, index=gene_names, columns=gene_names)


# ─── 4. Strict Direction ─────────────────────────────────────────────────────

def strict_direction(data: np.ndarray, lambd: float = 1.0) -> np.ndarray:
    """Zero the weaker direction of each (i,j)/(j,i) edge pair.

    Parameters
    ----------
    data : np.ndarray
        Square adjacency matrix.
    lambd : float
        0 = keep original; 1 = fully strict.
    """
    if lambd == 0:
        return data
    s = data.copy()
    s[np.abs(s) < np.abs(s.T)] = 0
    return (1 - lambd) * data + lambd * s


# ─── 5. Virtual Knockout ────────────────────────────────────────────────────

def virtual_knockout(
    wt_tensor: pd.DataFrame,
    ko_genes: List[str],
) -> pd.DataFrame:
    """Zero out the rows corresponding to ``ko_genes`` in the WT tensor.

    This implements the "default" KO method from scTenifoldKnk.
    """
    ko = wt_tensor.copy()
    ko.loc[ko_genes, :] = 0.0
    return ko


# ─── 6. Manifold Alignment ──────────────────────────────────────────────────

def manifold_alignment(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    d: int = MA_DIM,
    tol: float = 1e-8,
    verbose: bool = True,
) -> pd.DataFrame:
    """Laplacian eigenmaps manifold alignment (GPU ``torch.eigh``).

    Parameters
    ----------
    X_df, Y_df : pd.DataFrame
        (n_genes, n_genes) adjacency matrices for WT and KO.
    d : int
        Embedding dimension.
    tol : float
        Eigenvalue tolerance.

    Returns
    -------
    pd.DataFrame
        (2 * n_shared, d) embedding.  Index labels are "X_{gene}", "Y_{gene}".
    """
    shared = [g for g in X_df.index if g in Y_df.index]
    if not shared:
        raise ValueError("No shared genes between WT and KO tensors")

    X = X_df.loc[shared, shared].values.astype(np.float32)
    Y = Y_df.loc[shared, shared].values.astype(np.float32)
    n = len(shared)

    L = np.eye(n, dtype=np.float32)
    w_X, w_Y = X + 1, Y + 1
    w_XY = L * (0.9 * (np.sum(w_X) + np.sum(w_Y)) / (2 * n))

    W = -np.block([[w_X, w_XY], [w_XY.T, w_Y]])
    np.fill_diagonal(W, 0)
    np.fill_diagonal(W, -W.sum(axis=0))

    if verbose:
        _tlog(f"Eigendecomposition ({2*n}×{2*n}) on {DEVICE} ...")

    W_t = torch.from_numpy(W).to(DEVICE)
    evals, evecs = torch.linalg.eigh(W_t)
    evals, evecs = evals.cpu().numpy(), evecs.cpu().numpy()

    valid = evals >= tol
    evals, evecs = evals[valid], evecs[:, valid]
    evecs = evecs[:, np.argsort(evals)]
    d_actual = min(d, evecs.shape[1])

    x_lbl = [f"X_{g}" for g in shared]
    y_lbl = [f"Y_{g}" for g in shared]
    cols = [f"NLMA_{i+1}" for i in range(d_actual)]

    return pd.DataFrame(evecs[:, :d_actual], index=x_lbl + y_lbl, columns=cols)


# ─── 7. Differential Regulation ─────────────────────────────────────────────

def differential_regulation(
    ma_df: pd.DataFrame,
    n_ko_genes: int = 1,
    ko_genes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute differential regulation scores from manifold alignment.

    Parameters
    ----------
    ma_df : pd.DataFrame
        Output of :func:`manifold_alignment`.
    n_ko_genes : int
        Number of knocked-out genes (fallback rank-based exclusion).
    ko_genes : list of str or None
        Names of knocked-out genes.  When provided, these genes are
        excluded from the FC baseline by identity (not rank).

    Returns
    -------
    pd.DataFrame
        Columns: Gene, Distance, boxcox_transformed_distance, Z, FC,
        p-value, adjusted_p-value.  Sorted by p-value ascending.
    """
    all_names = ma_df.index.to_list()
    gene_names = [g[2:] for g in all_names if g.startswith("X_")]
    n = len(gene_names)

    if n * 2 != len(all_names):
        raise ValueError("Gene count mismatch in manifold alignment result")

    d_metrics = np.array([
        np.linalg.norm(ma_df.iloc[i, :].values - ma_df.iloc[i + n, :].values)
        for i in range(n)
    ])

    t_d = d_metrics.astype(float).copy()
    pos = d_metrics > 0
    try:
        if pos.any():
            t, maxlog = stats.boxcox(d_metrics[pos])
            t_d[pos] = -1 / np.array(t) if maxlog < 0 else np.array(t)
    except Exception:
        pass

    std_ = t_d.std()
    z = np.zeros_like(t_d) if std_ == 0 else (t_d - t_d.mean()) / std_

    if ko_genes is not None:
        ko_set = set(ko_genes)
        non_ko_mask = np.array([g not in ko_set for g in gene_names])
    else:
        sorted_idx = np.argsort(d_metrics)[::-1]
        non_ko_mask = np.ones(n, dtype=bool)
        non_ko_mask[sorted_idx[:n_ko_genes]] = False

    non_ko_d = d_metrics[non_ko_mask]
    expected = np.mean(non_ko_d ** 2) if len(non_ko_d) > 0 else 1.0
    FC = d_metrics ** 2 / expected if expected > 0 else np.zeros_like(d_metrics)

    p_vals = stats.chi2.sf(FC, df=1)
    p_adj = cal_fdr(p_vals)

    result = pd.DataFrame({
        "Gene": gene_names,
        "Distance": d_metrics,
        "boxcox_transformed_distance": t_d,
        "Z": z,
        "FC": FC,
        "p-value": p_vals,
        "adjusted_p-value": p_adj,
    })
    return result.sort_values("p-value", ascending=True)


# ─── High-level API ──────────────────────────────────────────────────────────

class scTenifoldKnkCUDA:
    """Single-sample virtual-knockout workflow (CUDA-accelerated).

    Parameters
    ----------
    data : pd.DataFrame
        genes × cells expression matrix.
    ko_genes : str or list of str
        Gene name(s) to knock out.
    n_nets : int
        Number of bootstrap PC networks.
    n_samp_cells : int or None
        Cells per subsample (None = all cells).
    n_comp : int
        PC components per gene regression.
    q : float
        Quantile threshold for edge pruning.
    strict_lambda : float
        Direction pruning strength (0 = off).
    ma_dim : int
        Manifold alignment dimension.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        ko_genes: List[str],
        n_nets: int = 10,
        n_samp_cells: Optional[int] = 500,
        n_comp: int = N_COMP,
        q: float = Q_THRESH,
        strict_lambda: float = STRICT_LAMBDA,
        ma_dim: int = MA_DIM,
        random_state: int = DEFAULT_RANDOM_SEED,
        qc_kws: Optional[dict] = None,
    ):
        self.data = data
        self.ko_genes = ko_genes if isinstance(ko_genes, list) else [ko_genes]
        self.n_nets = n_nets
        self.n_samp_cells = n_samp_cells
        self.n_comp = n_comp
        self.q = q
        self.strict_lambda = strict_lambda
        self.ma_dim = ma_dim
        self.random_state = random_state
        self.qc_kws = qc_kws or {}

        # Intermediate results
        self.qc_data: Optional[pd.DataFrame] = None
        self.networks: Optional[List[np.ndarray]] = None
        self.wt_tensor: Optional[pd.DataFrame] = None
        self.ko_tensor: Optional[pd.DataFrame] = None
        self.ma_result: Optional[pd.DataFrame] = None
        self.d_reg: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return differential regulation table."""
        t_total = time.perf_counter()

        # 1. QC
        _tlog("Step 1/6: Quality Control")
        t0 = time.perf_counter()
        self.qc_data = run_qc(self.data, **self.qc_kws)
        _tlog(f"  → {self.qc_data.shape[0]} genes × {self.qc_data.shape[1]} cells "
              f"({time.perf_counter()-t0:.1f}s)")

        # Verify KO genes survived QC
        missing = [g for g in self.ko_genes if g not in self.qc_data.index]
        if missing:
            raise ValueError(f"KO genes not found after QC: {missing}")

        # 2. PC networks
        _tlog("Step 2/6: PC Network Construction (CUDA)")
        t0 = time.perf_counter()
        self.networks = build_pc_networks(
            self.qc_data,
            n_nets=self.n_nets,
            n_samp_cells=self.n_samp_cells,
            n_comp=self.n_comp,
            q=self.q,
            random_state=self.random_state,
        )
        _tlog(f"  → {len(self.networks)} networks ({time.perf_counter()-t0:.1f}s)")

        # 3. Tensor decomposition
        _tlog("Step 3/6: Tensor Decomposition")
        t0 = time.perf_counter()
        self.wt_tensor = tensor_decompose_mean(
            self.networks, self.qc_data.index.to_numpy()
        )
        del self.networks
        _tlog(f"  → WT tensor {self.wt_tensor.shape} ({time.perf_counter()-t0:.1f}s)")

        # Apply strict_direction, zero diagonal
        wt_arr = strict_direction(self.wt_tensor.values, self.strict_lambda).T.copy()
        np.fill_diagonal(wt_arr, 0)
        self.wt_tensor = pd.DataFrame(
            wt_arr, index=self.wt_tensor.index, columns=self.wt_tensor.columns
        )

        # 4. Knockout
        _tlog(f"Step 4/6: Virtual Knockout of {self.ko_genes}")
        t0 = time.perf_counter()
        self.ko_tensor = virtual_knockout(self.wt_tensor, self.ko_genes)
        _tlog(f"  → KO tensor {self.ko_tensor.shape} ({time.perf_counter()-t0:.1f}s)")

        # 5. Manifold alignment
        _tlog("Step 5/6: Manifold Alignment (GPU)")
        t0 = time.perf_counter()
        self.ma_result = manifold_alignment(
            self.wt_tensor, self.ko_tensor, d=self.ma_dim
        )
        _tlog(f"  → {self.ma_result.shape} ({time.perf_counter()-t0:.1f}s)")

        # 6. Differential regulation
        _tlog("Step 6/6: Differential Regulation")
        t0 = time.perf_counter()
        self.d_reg = differential_regulation(
            self.ma_result,
            n_ko_genes=len(self.ko_genes),
            ko_genes=self.ko_genes,
        )
        _tlog(f"  → {self.d_reg.shape[0]} genes tested ({time.perf_counter()-t0:.1f}s)")

        _tlog(f"Total: {time.perf_counter()-t_total:.1f}s")
        return self.d_reg
