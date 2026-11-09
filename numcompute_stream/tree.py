import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth, min_samples_split=2, max_features=None,
                 buffer_size=500, criterion='gini'):
        if max_features is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError('max_depth must be a non-negative integer or None')
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError('min_samples_split must be greater than 2')
        if not isinstance(buffer_size, int) or buffer_size < 1:
            raise ValueError('buffer_size must be positive')
        if criterion not in ['gini', 'entropy']:
            raise ValueError('criterion must be "gini" or "entropy"')
        if max_features is not None and not (isinstance(max_features, int) and max_features > 0):
            raise ValueError('max_features must be a positive integer')

        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.max_features, self.buffer_size, self.criterion = max_features, buffer_size, criterion
        self.tree_ = None
        self.n_features_in_ = None
        self._X, self._Y = [], []
        self.buffer_count_ = 0
        self.n_retrains_ = 0

    def _impurity(self, counts):
        if self.criterion == 'gini':
            return gini(counts)
        return _entropy(counts)

    def _best_split(self, X, y, counts):
        """Return (feature, threshold, left_mask), or None without positive gain."""
        best_gain, best = 0.0, None
        parent_impurity = self._impurity(counts)
        feature_count = X.shape[1] if self.max_features is None else min(X.shape[1], self.max_features)
        for feature in range(feature_count):
            column = X[:, feature]
            missing = np.isnan(column)
            values = np.unique(column[~missing])
            if not len(values):
                continue

            sortable = np.where(missing, -np.inf, column)
            order = np.argsort(sortable, kind='stable')
            sorted_values = sortable[order]
            cumulative = np.cumsum(np.eye(len(self.classes_), dtype=int)[y[order]], axis=0)
            if missing.any():
                values = np.r_[-np.inf, values]
            positions = np.searchsorted(sorted_values, values, side='right')
            positions = positions[(positions > 0) & (positions < len(y))]
            if not len(positions):
                continue

            left_counts = cumulative[positions - 1]
            right_counts = counts - left_counts
            gains = parent_impurity - (
                positions * self._impurity(left_counts)
                + (len(y) - positions) * self._impurity(right_counts)
            ) / len(y)
            index = int(np.argmax(gains))
            if gains[index] > best_gain + 1e-12:
                threshold = sorted_values[positions[index] - 1]
                mask = missing | (column <= threshold)
                best_gain = gains[index]
                best = feature, threshold, mask
        return best

    def _build(self, X, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_))
        leaf = int(np.argmax(counts))
        if (
            np.count_nonzero(counts) == 1
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return leaf

        best = self._best_split(X, y, counts)
        if best is None:
            return leaf

        feature, threshold, mask = best
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return feature, threshold, left, right

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a non-empty 2D array")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError("feature count changed")

        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1D with one label per sample")

        self.classes_, encoded = np.unique(y, return_inverse=True)
        d = X.shape[1]
        k = self.max_features
        self._feature_count = d if k is None else max(1, int(np.sqrt(d))) if k == "sqrt" else max(1, int(np.log2(
            d))) if k == "log2" else min(d, k)
        self.n_features_in_ = d
        self.tree_ = self._build(X, encoded, 0)
        self._X, self._y, self.buffer_count_ = [], [], 0
        return self

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a non-empty 2D array")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError("feature count changed")

        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1D with one label per sample")

        self.n_features_in_ = X.shape[1]
        self._X.append(X.copy())
        self._y.append(y.copy())
        self.buffer_count_ += len(X)
        if self.buffer_count_ >= self.buffer_size:
            self.fit(np.concatenate(self._X), np.concatenate(self._y))
            self.n_retrains_ += 1
        return self

    def _visit(self, node, indices, X, out):
        if isinstance(node, int):
            out[indices] = node
            return

        feature, threshold, left, right = node
        mask = (
                np.isnan(X[indices, feature])
                | (X[indices, feature] <= threshold)
        )

        self._visit(left, indices[mask], X, out)
        self._visit(right, indices[~mask], X, out)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a non-empty 2D array")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError("feature count changed")
        if self.tree_ is None:
            raise ValueError("tree has not trained; provide at least buffer_size samples")

        out = np.empty(len(X), dtype=int)

        self._visit(self.tree_, np.arange(len(X)), X, out)

        return self.classes_[out]

def gini(counts):
    """Calculate Gini impurity from class counts along the last axis."""
    counts = np.asarray(counts)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts, dtype=float), where=total != 0)
    return 1 - (p * p).sum(axis=-1)

def _entropy(counts):
    """Calculate entropy from class counts along the last axis."""
    counts = np.asarray(counts)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts, dtype=float), where=total != 0)
    return -(p * np.log2(np.where(p > 0, p, 1))).sum(axis=-1)
