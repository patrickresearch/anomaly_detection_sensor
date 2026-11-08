# anomaly_detection_sensor

Comparing five anomaly detectors on nine labeled sensor time series from the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB), evaluated at event level against a chance baseline.

![hero](docs/hero.png)

| Detector | Idea |
|---|---|
| `zscore` | deviation from trailing rolling mean (baseline) |
| `iforest` | Isolation Forest on trailing window statistics |
| `matrix_profile` | z-normalized distance to nearest normal training subsequence (stumpy AB-join) |
| `matrix_profile_raw` | same without z-normalization, sensitive to level and amplitude shifts |
| `autoencoder` | reconstruction error of a 1D-conv autoencoder trained on normal windows |

All detectors share one `fit/score` interface ([src/detectors.py](src/detectors.py)) and one thresholding rule: a fixed score quantile on normal validation points.

## Evaluation

- chronological split 40/20/40, labeled windows removed from training
- weekly time-of-week profile removed for series with a weekly cycle
- event-level precision, recall and F1; alarms within one context length after a window are not counted as false positives
- chance F1 and p-value from random circular shifts of each detector's alarm sequence
- five seeds for stochastic detectors, F1 over a range of threshold quantiles

## Usage

```bash
uv sync
uv run python -m src.run
uv run jupyter nbconvert --to html --execute notebooks/anomaly_comparison.ipynb --output-dir docs --output index.html
```
