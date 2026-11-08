import math

import numpy as np
import pandas as pd

PROFILES = {"standard": (1.0, 0.11, 1.0), "reward_low_FP_rate": (1.0, 0.22, 1.0), "reward_low_FN_rate": (1.0, 0.11, 2.0)}


def scaled_sigmoid(x: float) -> float:
    return -1.0 if x > 3.0 else 2 / (1 + math.exp(5 * x)) - 1.0


def probation(n: int, percent: float = 0.15) -> int:
    return min(math.floor(percent * n), percent * 5000)


def raw_score(t: pd.DatetimeIndex, alarms: np.ndarray, windows: list, profile: str, skip: int = 0) -> tuple[float, int]:
    tp_w, fp_w, fn_w = PROFILES[profile]
    max_tp = scaled_sigmoid(-1.0)
    bounds = []
    for a, b in windows:
        if b < t[0] or a > t[-1]:
            continue
        left, right = t.searchsorted(max(a, t[0])), t.searchsorted(min(b, t[-1]), side="right") - 1
        bounds.append((left, right))
    fp, best, prev = 0.0, {}, None
    hits = np.flatnonzero(alarms)
    for j, (left, right) in enumerate(bounds):
        if right >= skip:
            best[j] = -fn_w
    for i in hits:
        if i < skip:
            continue
        j = next((j for j, (l, r) in enumerate(bounds) if l <= i <= r), None)
        if j is not None:
            l, r = bounds[j]
            best[j] = max(best[j], scaled_sigmoid(-(r - i + 1) / (r - l + 1)) * tp_w / max_tp)
            continue
        before = [(l, r) for l, r in bounds if r < i]
        if not before:
            fp += -1.0 * fp_w
        else:
            l, r = before[-1]
            fp += scaled_sigmoid(abs(r - i) / max(r - l, 1)) * fp_w
    return fp + sum(best.values()), len(best)


def normalized(parts: list[tuple[float, int]], profile: str) -> float:
    tp_w, _, fn_w = PROFILES[profile]
    score, windows = sum(p[0] for p in parts), sum(p[1] for p in parts)
    null, perfect = -fn_w * windows, tp_w * windows
    return 100 * (score - null) / (perfect - null) if windows else float("nan")
