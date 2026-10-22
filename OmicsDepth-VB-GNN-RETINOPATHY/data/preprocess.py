"""Preprocessing (SI, Section 2: retinal and optic nerve data mapping).

Pipeline used for the real-data analysis:
  1. aggregate scRNA-seq cells to donor x cell-type pseudobulk profiles, so that
     the donor (not the cell) is the unit of replication;
  2. log-transform;
  3. residualise against technical batch and biological covariates
     (age at death, sex, post-mortem interval);
  4. standardise features;
  5. estimate a positive-definite feature covariance Sigma.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler


def pseudobulk_aggregate(counts: np.ndarray, donor_ids, cell_types=None, normalize_cpm: bool = True):
    """Sum single-cell counts per donor (and per cell type, if given).

    Args:
        counts: [n_cells, n_genes] raw counts.
        donor_ids: length-n_cells donor labels.
        cell_types: optional length-n_cells cell-type labels; if given, the output
            has one block of columns per cell type ("<cell_type>|<gene index>").

    Returns:
        (matrix [n_donors, n_features], donor_order, column_labels)
    """
    counts = np.asarray(counts, dtype=float)
    donor_ids = np.asarray(donor_ids)
    donors = np.unique(donor_ids)
    groups = [None] if cell_types is None else list(np.unique(np.asarray(cell_types)))
    blocks, labels = [], []
    for ct in groups:
        sel_ct = np.ones(len(donor_ids), bool) if ct is None else (np.asarray(cell_types) == ct)
        block = np.vstack([counts[sel_ct & (donor_ids == d)].sum(axis=0) for d in donors])
        if normalize_cpm:
            lib = block.sum(axis=1, keepdims=True)
            block = np.divide(block, lib, out=np.zeros_like(block), where=lib > 0) * 1e6
        blocks.append(block)
        labels += [f"{ct}|{g}" if ct is not None else str(g) for g in range(counts.shape[1])]
    return np.hstack(blocks), donors, labels


def log_transform(X: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    return np.log2(np.asarray(X, dtype=float) + pseudocount)


def residualize(X: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Remove the linear effect of covariates (with intercept) by least squares."""
    X = np.asarray(X, dtype=float)
    C = np.column_stack([np.ones(len(X)), np.asarray(covariates, dtype=float)])
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ beta + X.mean(axis=0)


def standardize_omics_data(X: np.ndarray) -> np.ndarray:
    """Standardize multi-omic features to zero mean and unit variance."""
    return StandardScaler().fit_transform(X)


def shrinkage_covariance(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf covariance estimate; positive definite, as required for Sigma in S^d_{++}."""
    return LedoitWolf().fit(np.asarray(X, dtype=float)).covariance_


def preprocess_pipeline(X: np.ndarray, covariates: np.ndarray, already_log: bool = False):
    """log-transform (optional) -> residualise -> standardise -> (X, Sigma)."""
    Xp = X if already_log else log_transform(X)
    Xp = standardize_omics_data(residualize(Xp, covariates))
    return Xp, shrinkage_covariance(Xp)
