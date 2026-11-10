import pickle
from time import perf_counter
from .metrics import ClassificationMetrics, accuracy


class StreamTrainer:
    """score_chunk evaluates without training; fit_chunk trains without scoring.
    """

    def __init__(self, model=None, pipeline=None, window_size=None):
        if (model is None) == (pipeline is None):
            raise ValueError('provide exactly one model or pipeline')
        self.model = model if model is not None else pipeline
        self.metrics = ClassificationMetrics(window_size)
        self.logs = []
        self.chunks_ = 0

    def fit_chunk(self, X, y):
        start = perf_counter()
        self.model.partial_fit(X, y)
        self.chunks_ += 1
        self.logs.append(
            dict(
                event='fit',
                chunk=self.chunks_,
                seconds=perf_counter() - start,
                memory_bytes=len(
                    pickle.dumps(
                        self.model)),
                cumulative_accuracy=self.metrics.result()['accuracy']))
        return self

    def score_chunk(self, X, y):
        pred = self.model.predict(X)
        self.metrics.update(y, pred)
        result = self.metrics.result()
        self.logs.append(
            dict(
                event='score',
                chunk=self.chunks_ + 1,
                accuracy=accuracy(
                    y,
                    pred),
                cumulative_accuracy=result['accuracy'],
                memory_bytes=len(
                    pickle.dumps(
                        self.model))))
        return result
