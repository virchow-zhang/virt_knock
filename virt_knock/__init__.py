"""virt_knock — CUDA-Native Single-Cell Virtual Gene Knockout

Implements the scTenifoldKnk algorithm (Osorio et al., Patterns, 2022)
with native CUDA acceleration via PyTorch.

Pipeline:
    1. QC filtering
    2. PC network construction (randomized SVD on GPU)
    3. Tensor decomposition (bootstrap mean)
    4. Virtual knockout (default or propagation)
    5. Manifold alignment (GPU Laplacian eigenmaps)
    6. Differential regulation (chi-square + FDR)
    7. Enrichment analysis (GSEA + ORA, optional)
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .core import (
    run_qc,
    build_pc_networks,
    tensor_decompose_mean,
    strict_direction,
    virtual_knockout,
    manifold_alignment,
    differential_regulation,
    scTenifoldKnkCUDA,
)
from .enrichment import (
    run_enrichment_all,
    run_gsea_prerank,
    run_ora_enrichr,
)

__version__ = "0.2.0"
__all__ = [
    "run_qc",
    "build_pc_networks",
    "tensor_decompose_mean",
    "strict_direction",
    "virtual_knockout",
    "manifold_alignment",
    "differential_regulation",
    "scTenifoldKnkCUDA",
    "run_enrichment_all",
    "run_gsea_prerank",
    "run_ora_enrichr",
]
