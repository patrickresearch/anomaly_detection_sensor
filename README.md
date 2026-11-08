# anomaly_detection_sensor

Comparing anomaly detectors on labeled sensor time series: five univariate detectors on nine series from the
[Numenta Anomaly Benchmark](https://github.com/numenta/NAB) and three multivariate detectors with per-sensor explanations on
[SKAB](https://github.com/waico/SKAB). Evaluated at event level against a chance baseline, with a streaming variant and an
interactive threshold app.

![hero](docs/hero.png)

## Univariate (NAB)

| Detector | Idea |
|---|---|
| `zscore` | deviation from trailing rolling mean (baseline) |
| `iforest` | Isolation Forest on trailing window statistics |
| `matrix_profile` | z-normalized distance to nearest normal training subsequence (stumpy AB-join) |
| `matrix_profile_raw` | same without z-normalization, sensitive to level and amplitude shifts |
| `autoencoder` | reconstruction error of a 1D-conv autoencoder trained on normal windows |

- chronological split 40/20/40, labeled windows removed from training
- weekly time-of-week profile removed for series with a weekly cycle
- event-level precision, recall and F1; alarms within one context length after a window are not counted as false positives
- chance F1 and p-value from random circular shifts of each detector's alarm sequence
- five seeds for stochastic detectors, F1 over a range of threshold quantiles

## Multivariate with explanation (SKAB)

| Detector | Contribution of each sensor |
|---|---|
| `zscore` | squared standardized value |
| `pca` | complete decomposition of the combined T² + SPE index (Alcala & Qin 2011) |
| `autoencoder` | per-channel reconstruction error of a multichannel conv autoencoder |

Contributions sum to the anomaly score, so the explanation for an alarm is the share of each sensor. It is validated on faults
injected into single sensors of the anomaly-free recording and checked for plausibility against SKAB's documented fault types.

## Streaming

![streaming](docs/streaming.gif)

Every detector scores point *t* from the trailing window ending at *t*, so a fixed buffer reproduces the batch scores exactly
([src/streaming.py](src/streaming.py)). The GIF shows the multichannel autoencoder on a rotor-imbalance experiment.

## App

[app/streamlit_app.py](app/streamlit_app.py) lets you move the threshold and see alarms, false alarms, delay and, for SKAB, the
sensor explanation update. It reads the precomputed files in `results/` and does not need torch.

## Usage

```bash
uv sync
uv run python -m src.run           # NAB
uv run python -m src.run_skab      # SKAB
uv run python -m src.animate       # docs/streaming.gif
uv run pytest
uv run jupyter nbconvert --to html --execute notebooks/anomaly_comparison.ipynb --output-dir docs --output index.html
uv run streamlit run app/streamlit_app.py
```

Deploying the app on Streamlit Community Cloud: create an app from this repository with main file `app/streamlit_app.py`
and Python 3.12. Community Cloud uses `app/requirements.txt`, because a dependency file next to the entrypoint takes precedence
over `uv.lock` in the repository root.
