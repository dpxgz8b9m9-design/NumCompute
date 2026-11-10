import json
import sys
from pathlib import Path
from time import perf_counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from numcompute_stream import DecisionTreeClassifier


def loop_accuracy(y, p):
    return sum(int(a == b) for a, b in zip(y, p))/len(y)


def run():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 100000)
    pred = rng.integers(0, 2, len(y))
    timings = {}
    for name, fn in [('loop', loop_accuracy), ('vectorised', lambda a,b: float(np.mean(a == b)))]:
        durations = []
        for _ in range(5):
            start = perf_counter()
            value = fn(y, pred)
            durations.append(perf_counter()-start)
        timings[name] = dict(seconds_median=float(np.median(durations)), accuracy=value)
    X = rng.normal(size=(2400, 4))
    target = (X[:, 0]+.4*X[:, 1] > 0).astype(int)
    target[1600:] = 1-target[1600:]  # abrupt concept drift
    results = []
    # Shared 600-row warm-up and identical 1,800 evaluated samples for fairness.
    for size in [100, 300, 600]:
        model = DecisionTreeClassifier(max_depth=5, buffer_size=size)
        model.fit(X[:600], target[:600])
        correct, seconds, history = 0, 0., []
        for start in range(600, len(X), 100):
            chunk, truth = X[start:start+100], target[start:start+100]
            prediction = model.predict(chunk)
            correct += int(np.sum(prediction == truth))
            history.append(float(np.mean(prediction == truth)))
            tick = perf_counter()
            model.partial_fit(chunk, truth)
            seconds += perf_counter()-tick
        results.append(dict(buffer_size=size, accuracy=correct/1800,
                            update_seconds=seconds, retrains=model.n_retrains_, chunk_accuracy=history))
    result = dict(seed=42, samples=2400, evaluated_samples=1800, warmup_samples=600,
                  loop_vs_vectorised=timings, buffers=results)
    out = Path(__file__).with_name('stream_results.json')
    out.write_text(json.dumps(result, indent=2)+'\n')
    rows = ['# Streaming benchmark', '', 'Seed 42; 600 common warm-up rows; 1,800 identical prequential evaluation rows.',
            'Concept drift starts at row 1,600. Timing includes all partial_fit calls, excluding initial fit.', '',
            '| Buffer | Accuracy | Update time (s) | Retrains |', '|---:|---:|---:|---:|']
    rows += [f"| {r['buffer_size']} | {r['accuracy']:.4f} | {r['update_seconds']:.6f} | {r['retrains']} |" for r in results]
    rows += ['', 'Accuracy kernel: 100,000 samples, median of five runs. Equivalent outputs verified.', '']
    assert timings['loop']['accuracy'] == timings['vectorised']['accuracy']
    rows += [f"- {name}: {value['seconds_median']:.6f} s" for name, value in timings.items()]
    Path(__file__).with_name('stream_results.md').write_text('\n'.join(rows)+'\n')
    print('\n'.join(rows))
    return result


if __name__ == '__main__':
    run()
