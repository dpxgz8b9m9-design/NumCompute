# Streaming benchmark

Seed 42; 600 common warm-up rows; 1,800 identical prequential evaluation rows.
Concept drift starts at row 1,600. Timing includes all partial_fit calls, excluding initial fit.

| Buffer | Accuracy | Update time (s) | Retrains |
|---:|---:|---:|---:|
| 100 | 0.8656 | 0.014586 | 18 |
| 300 | 0.8189 | 0.013142 | 6 |
| 600 | 0.5672 | 0.007708 | 3 |

Accuracy kernel: 100,000 samples, median of five runs. Equivalent outputs verified.

- loop: 0.015828 s
- vectorised: 0.000051 s
