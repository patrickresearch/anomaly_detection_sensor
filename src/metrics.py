import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def runs(idx: np.ndarray, gap: int) -> int:
    return 0 if len(idx) == 0 else 1 + int((np.diff(idx) > gap).sum())


def evaluate(
    t: pd.DatetimeIndex,
    alarms: np.ndarray,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    points: list[pd.Timestamp],
    gap: int,
) -> dict:
    in_window = np.zeros(len(t), dtype=bool)
    tp, delays = 0, []
    for start, end in windows:
        mask = (t >= start) & (t <= end)
        if not mask.any():
            continue
        in_window |= mask
        hits = np.flatnonzero(mask & alarms)
        if len(hits):
            tp += 1
            point = next(p for p in points if start <= p <= end)
            delays.append((t[hits[0]] - point) / pd.Timedelta("1h"))
    events = sum(((t >= a) & (t <= b)).any() for a, b in windows)
    fp = runs(np.flatnonzero(alarms & ~in_window), gap)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / events if events else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "events": events,
        "tp": tp,
        "fp": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "delay_h": float(np.median(delays)) if delays else np.nan,
        "point_f1": f1_score(in_window, alarms, zero_division=0),
    }
