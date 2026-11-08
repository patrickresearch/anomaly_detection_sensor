import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def runs(idx: np.ndarray, gap: int) -> int:
    return 0 if len(idx) == 0 else 1 + int((np.diff(idx) > gap).sum())


def regions(t: pd.DatetimeIndex, windows: list, grace: pd.Timedelta):
    masks = [(a, b, m) for a, b in windows if (m := (t >= a) & (t <= b)).any()]
    in_window = np.zeros(len(t), dtype=bool)
    for _, _, m in masks:
        in_window |= m
    after = np.zeros(len(t), dtype=bool)
    for _, b in windows:
        after |= (t > b) & (t <= b + grace)
    return masks, in_window, after & ~in_window


def counts(alarms: np.ndarray, masks: list, outside: np.ndarray, gap: int) -> tuple[int, int]:
    tp = sum(bool((m & alarms).any()) for _, _, m in masks)
    return tp, runs(np.flatnonzero(alarms & outside), gap)


def prf(tp: int, fp: int, events: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / events if events else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def chance(alarms, masks, outside, gap, events, n, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, len(alarms), n)
    return np.array([prf(*counts(np.roll(alarms, k), masks, outside, gap), events)[2] for k in shifts])


def evaluate(
    t: pd.DatetimeIndex,
    alarms: np.ndarray,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    points: list[pd.Timestamp],
    gap: int,
    grace: pd.Timedelta,
    n_null: int = 0,
    seed: int = 0,
) -> dict:
    masks, in_window, after = regions(t, windows, grace)
    outside = ~in_window & ~after
    events = len(masks)
    tp, fp = counts(alarms, masks, outside, gap)
    precision, recall, f1 = prf(tp, fp, events)
    delays = []
    for a, b, m in masks:
        hits = np.flatnonzero(m & alarms)
        if len(hits):
            point = next(p for p in points if a <= p <= b)
            delays.append((t[hits[0]] - point) / pd.Timedelta("1h"))
    null = chance(alarms, masks, outside, gap, events, n_null, seed) if n_null else np.array([np.nan])
    return {
        "events": events,
        "tp": tp,
        "fp": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "chance_f1": float(null.mean()),
        "p_value": float((1 + (null >= f1).sum()) / (1 + len(null))) if n_null else np.nan,
        "delay_h": float(np.median(delays)) if delays else np.nan,
        "point_f1": f1_score(in_window, alarms, sample_weight=(~after).astype(float), zero_division=0),
    }
