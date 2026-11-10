"""Streaming pipeline with raw-buffer replay to keep tree coordinates consistent."""
from copy import deepcopy
import numpy as np
from numcompute.pipeline import Pipeline as BatchPipeline
from .tree import DecisionTreeClassifier


class Pipeline(BatchPipeline):
    """Update transformers each chunk; snapshot them whenever the tree retrains.
    """
    def __init__(self, steps):
        super().__init__(steps)
        self._raw_X, self._raw_y = [], []
        self._prediction_steps = None

    def fit(self, X, y=None):
        super().fit(X, y)
        self._raw_X, self._raw_y = [], []
        self._prediction_steps = deepcopy(self.steps[:-1])
        return self

    def partial_fit(self, X, y=None):
        if any(not callable(getattr(step, 'partial_fit', None)) for _, step in self.steps):
            raise ValueError('all steps must implement partial_fit')
        Xt = X
        for _, step in self.steps[:-1]:
            step.partial_fit(Xt)
            Xt = step.transform(Xt)
        model = self.steps[-1][1]
        if isinstance(model, DecisionTreeClassifier):
            from ._validation import labels
            raw = np.asarray(X)
            if raw.ndim != 2 or not all(raw.shape):
                raise ValueError('X must be non-empty 2D')
            y = labels(y, len(raw))
            if self._raw_X and raw.shape[1] != self._raw_X[0].shape[1]:
                raise ValueError('feature count changed')
            self._raw_X.append(raw.copy())
            self._raw_y.append(y.copy())
            model.buffer_count_ = sum(len(a) for a in self._raw_X)
            if model.buffer_count_ >= model.buffer_size:
                buffered = np.concatenate(self._raw_X)
                for _, step in self.steps[:-1]:
                    buffered = step.transform(buffered)
                model.fit(buffered, np.concatenate(self._raw_y))
                model.n_retrains_ += 1
                self._raw_X, self._raw_y = [], []
                self._prediction_steps = deepcopy(self.steps[:-1])
        else:
            model.partial_fit(Xt, y) if y is not None else model.partial_fit(Xt)
            self._prediction_steps = deepcopy(self.steps[:-1])
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        Xt = X
        for _, step in self._prediction_steps or []:
            Xt = step.transform(Xt)
        return self.steps[-1][1].predict(Xt)
