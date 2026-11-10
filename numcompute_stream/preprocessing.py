import numpy as np
from ._validation import matrix


class StandardScaler:
    def __init__(self):
        self.mean_ = None

    def fit(self, X):
        self.mean_ = None
        return self.partial_fit(X)

    def partial_fit(self, X):
        X = matrix(X, None if self.mean_ is None else len(self.mean_))
        if self.mean_ is None:
            self.count_ = np.zeros(X.shape[1], dtype=int)
            self.mean_ = np.zeros(X.shape[1])
            self._m2 = np.zeros(X.shape[1])

        count = (~np.isnan(X)).sum(axis=0)
        mean = np.divide(
            np.nansum(
                X, axis=0), count, out=np.zeros(
                X.shape[1]), where=count > 0)
        m2 = np.nansum((X - mean) ** 2, axis=0)
        total = self.count_ + count
        delta = mean - self.mean_
        self._m2 += m2 + np.divide(delta ** 2 * self.count_ *
                                   count, total, out=np.zeros_like(mean), where=total > 0)
        self.mean_ += np.divide(delta * count, total,
                                out=np.zeros_like(mean), where=total > 0)
        self.count_ = total
        self.var_ = np.divide(
            self._m2,
            total,
            out=np.zeros_like(mean),
            where=total > 0)
        self.scale_ = np.sqrt(self.var_)
        self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("fit scaler first")
        return (matrix(X, len(self.mean_)) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler(StandardScaler):
    def __init__(self, feature_range=(0, 1)):
        super().__init__()
        if len(feature_range) != 2 or feature_range[0] >= feature_range[1]:
            raise ValueError("invalid feature_range")
        self.feature_range = feature_range

    def partial_fit(self, X):
        X = matrix(X, None if self.mean_ is None else len(self.mean_))
        first = self.mean_ is None
        super().partial_fit(X)
        lo = np.min(np.where(np.isnan(X), np.inf, X), axis=0)
        hi = np.max(np.where(np.isnan(X), -np.inf, X), axis=0)
        self.data_min_ = lo if first else np.minimum(self.data_min_, lo)
        self.data_max_ = hi if first else np.maximum(self.data_max_, hi)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("fit scaler first")
        X = matrix(X, len(self.mean_))
        lo = np.where(self.count_ > 0, self.data_min_, 0)
        hi = np.where(self.count_ > 0, self.data_max_, 0)
        span = np.where(hi > lo, hi - lo, 1)
        a, b = self.feature_range
        return (X - lo) / span * (b - a) + a


class SimpleImputer(StandardScaler):
    def __init__(self, strategy="mean", fill_value=0):
        super().__init__()
        if strategy not in ("mean", "constant") or not np.isfinite(fill_value):
            raise ValueError("invalid imputer settings")
        self.strategy, self.fill_value = strategy, fill_value

    def partial_fit(self, X):
        super().partial_fit(X)
        self.statistics_ = np.where(
            self.count_ > 0,
            self.mean_,
            self.fill_value) if self.strategy == "mean" else np.full(
            len(
                self.mean_),
            self.fill_value)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("fit imputer first")
        X = matrix(X, len(self.mean_))
        return np.where(np.isnan(X), self.statistics_, X)


Imputer = SimpleImputer


class OneHotEncoder(StandardScaler):
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        self.categories_ = None
        return self.partial_fit(X)

    def _validate(self, X):
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or not all(X.shape):
            raise ValueError("X must be non-empty and 2D array")
        if self.categories_ is not None and X.shape[1] != len(
                self.categories_):
            raise ValueError("feature count changed")
        return X

    def partial_fit(self, X):
        X = self._validate(X)
        if self.categories_ is None:
            self.categories_ = [[] for _ in range(X.shape[1])]
        for j, cats in enumerate(self.categories_):
            for v in X[:, j]:
                if v is not None and v == v and v not in cats:
                    cats.append(v)
        return self

    def transform(self, X):
        if self.categories_ is None:
            raise ValueError("fit encoder first")
        X = self._validate(X)
        return np.hstack([(X[:,
                             j:j + 1] == np.asarray(cats,
                           dtype=object)).astype(float) for j,
                          cats in enumerate(self.categories_)])

    def fit_transform(self, X):
        return self.fit(X).transform(X)
