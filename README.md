# anomaly_detection_sensor

Comparing four anomaly detectors on labeled sensor time series from the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB), evaluated at event level.

![hero](docs/hero.png)

| Detector | Idea |
|---|---|
| `zscore` | deviation from trailing rolling mean (baseline) |
| `iforest` | Isolation Forest on trailing window statistics |
| `matrix_profile` | distance to nearest normal training subsequence (stumpy AB-join) |
| `autoencoder` | reconstruction error of a 1D-conv autoencoder trained on normal windows |

All detectors share one `fit/score` interface ([src/detectors.py](src/detectors.py)) and one thresholding rule: a fixed score quantile on normal validation points.

## Usage

```bash
uv sync
uv run python -m src.run
uv run jupyter nbconvert --to html --execute notebooks/anomaly_comparison.ipynb --output-dir docs --output index.html
```
