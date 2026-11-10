"""Column moments plus exact retained-history quantiles/histograms."""
import warnings
import numpy as np
from .preprocessing import StandardScaler
from ._validation import matrix


class StreamingStats:
    def __init__(self, window_size=None):
        if window_size is not None and (
                not isinstance(
                    window_size,
                    int) or window_size < 1):
            raise ValueError('window_size must be positive')
        self.window_size = window_size
        self.reset()

    def reset(self):
        self._data = None
        self._moments = StandardScaler()
        return self

    def update_stats(self, X_chunk):
        X = matrix(
            X_chunk,
            None if self._data is None else self._data.shape[1])
        self._data = X.copy() if self._data is None else np.concatenate((self._data, X))
        if self.window_size is not None:
            self._data = self._data[-self.window_size:]
            self._moments.fit(self._data)
        else:
            self._moments.partial_fit(X)
        return self

    update = update_stats
    partial_fit = update_stats

    def result(self, q=(.25, .5, .75), bins=10, range_bounds=None):
        if self._data is None:
            raise ValueError('update statistics first')
        q = np.asarray(q)
        if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError('quantiles must lie in [0,1]')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            quantiles = np.nanquantile(self._data, q, axis=0)
        hist = [np.histogram(col[~np.isnan(col)], bins=bins,
                             range=range_bounds) for col in self._data.T]
        m = self._moments
        return dict(
            count=m.count_.copy(),
            mean=np.where(
                m.count_ > 0,
                m.mean_,
                np.nan),
            variance=np.where(
                m.count_ > 0,
                m.var_,
                np.nan),
            quantiles=quantiles,
            histograms=hist)
