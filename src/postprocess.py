import numpy as np
import pandas as pd


def smooth(score: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return score
    return pd.Series(score).ewm(span=span, adjust=False, ignore_na=True).mean().to_numpy()


def debounce(alarms: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return alarms
    c = np.concatenate([[0], np.cumsum(alarms)])
    out = np.zeros_like(alarms)
    out[k - 1:] = (c[k:] - c[:-k]) == k
    return out


def calibrate(score: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(reference[np.isfinite(reference)])
    out = np.searchsorted(ref, score, side="right") / len(ref)
    return np.where(np.isfinite(score), out, np.nan)


def ensemble(scores: dict[str, np.ndarray], reference: np.ndarray) -> np.ndarray:
    return np.mean([calibrate(s, s[reference]) for s in scores.values()], axis=0)


def alarms_for(score: np.ndarray, val: np.ndarray, q: float, span: int, k: int) -> np.ndarray:
    s = smooth(score, span)
    thr = np.nanquantile(s[val], q)
    return debounce(np.nan_to_num(s, nan=-np.inf) > thr, k)
