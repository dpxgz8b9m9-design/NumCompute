import numpy as np
from ._validation import labels


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = labels(y_pred, len(y_true))
    labels(y_true, len(y_pred))
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


class ClassificationMetrics:
    def __init__(self, window_size=None):
        if window_size is not None and (
                not isinstance(
                    window_size,
                    int) or window_size < 1):
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.reset()

    def reset(self):
        self._true, self._pred, self._scores = [], [], []
        self._has_scores = None
        return self

    def update(self, y_true, y_pred, y_score=None):
        y_true = np.asarray(y_true)
        y_true = labels(y_true, len(y_true))
        y_pred = labels(y_pred, len(y_true))
        if not len(y_true):
            return self
        has_scores = y_score is not None
        if self._has_scores is not None and has_scores != self._has_scores:
            raise ValueError("provide scores consistently")
        if has_scores:
            scores = np.asarray(y_score, dtype=float)
            if scores.shape != y_true.shape or not np.isfinite(
                    scores).all() or not np.isin(y_true, [0, 1]).all():
                raise ValueError(
                    "AUC requires binary 0/1 labels and finite 1D scores")
        self._has_scores = has_scores
        self._true.extend(y_true.tolist())
        self._pred.extend(y_pred.tolist())
        if has_scores:
            self._scores.extend(scores.tolist())
        if self.window_size is not None:
            self._true = self._true[-self.window_size:]
            self._pred = self._pred[-self.window_size:]
            self._scores = self._scores[-self.window_size:]
        return self

    def result(self):
        t, p = np.asarray(self._true), np.asarray(self._pred)
        classes = np.unique(np.concatenate((t, p)))
        k = len(classes)
        cm = np.zeros((k, k), dtype=int)
        if k:
            np.add.at(
                cm, (np.searchsorted(
                    classes, t), np.searchsorted(
                    classes, p)), 1)
        tp = np.diag(cm).astype(float)
        precision = np.divide(
            tp, cm.sum(
                axis=0), out=np.zeros(k), where=cm.sum(
                axis=0) > 0)
        recall = np.divide(
            tp, cm.sum(
                axis=1), out=np.zeros(k), where=cm.sum(
                axis=1) > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(k),
            where=precision + recall > 0)
        auc_value = float('nan')
        if self._has_scores and np.any(t == 1) and np.any(t == 0):
            scores = np.asarray(self._scores)
            order = np.argsort(scores, kind='stable')
            sorted_scores = scores[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
            ends = np.r_[starts[1:], len(scores)]
            ranks = np.empty(len(scores), dtype=float)
            ranks[order] = np.repeat((starts + 1 + ends) / 2, ends - starts)
            npos, nneg = np.sum(t == 1), np.sum(t == 0)
            auc_value = float(
                (ranks[t == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
        return dict(
            accuracy=accuracy(
                t,
                p),
            precision=float(
                precision.mean()) if k else 0.,
            recall=float(
                recall.mean()) if k else 0.,
            f1=float(
                f1.mean()) if k else 0.,
            confusion_matrix=cm,
            classes=classes,
            auc=auc_value,
            n_samples=len(t))


StreamingMetrics = ClassificationMetrics
