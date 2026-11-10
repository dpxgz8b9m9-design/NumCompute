import numpy as np


def matrix(X, n_features=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
        raise ValueError("X must be a non-empty 2D array")
    if np.isinf(X).any():
        raise ValueError("X must not contain infinity")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("feature count changed")
    return X

def labels(y, n):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n:
        raise ValueError("y must be 1D with one label per sample")
    if any(v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)) for v in y):
        raise ValueError("labels must not be missing or infinite")
    return y