import matplotlib.pyplot as plt


def _finish(fig, ax, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax


def plot_metric_over_time(
        metric_values,
        title='Streaming accuracy',
        ylabel='Accuracy',
        *,
        save_path=None,
        show=False):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(metric_values) + 1), metric_values, marker='o')
    ax.set(xlabel='Chunk', ylabel=ylabel, title=title)
    return _finish(fig, ax, save_path, show)


def compare_models(
        metric1,
        metric2,
        labels=(
            'Model 1',
            'Model 2'),
    *,
    save_path=None,
        show=False):
    if len(labels) != 2:
        raise ValueError('provide two labels')
    fig, ax = plt.subplots()
    for values, label in zip((metric1, metric2), labels):
        ax.plot(range(1, len(values) + 1), values, label=label)
    ax.set(xlabel='Chunk', ylabel='Accuracy',
           title='Streaming model comparison')
    ax.legend()
    return _finish(fig, ax, save_path, show)


def plot_predictions_vs_ground_truth(
        y_true,
        y_pred,
        *,
        save_path=None,
        show=False):
    if len(y_true) != len(y_pred):
        raise ValueError('length mismatch')
    fig, ax = plt.subplots()
    ax.plot(y_true, 'o', label='Ground truth')
    ax.plot(y_pred, 'x', label='Prediction')
    ax.set(xlabel='Sample', ylabel='Class', title='Latest chunk')
    ax.legend()
    return _finish(fig, ax, save_path, show)
