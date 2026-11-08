import numpy as np
import pandas as pd


def robust_z(score: np.ndarray, start: int, history: int, avg: int = 1, min_periods: int = 50) -> np.ndarray:
    s = pd.Series(np.where(np.arange(len(score)) >= start, score, np.nan))
    past = s.shift(1).rolling(history, min_periods=min_periods)
    med = past.median()
    mad = (s.shift(1) - med).abs().rolling(history, min_periods=min_periods).median() * 1.4826
    scale = mad.where(mad > 0, past.std()).where(lambda v: v > 0, 1e-12)
    recent = s.rolling(avg, min_periods=1).mean() if avg > 1 else s
    return ((recent - med) / scale).to_numpy()


def refractory(alarms: np.ndarray, period: int) -> np.ndarray:
    if period <= 0:
        return alarms
    idx = np.flatnonzero(alarms)
    out = np.zeros_like(alarms)
    if len(idx):
        out[idx[np.r_[True, np.diff(idx) > period]]] = True
    return out
