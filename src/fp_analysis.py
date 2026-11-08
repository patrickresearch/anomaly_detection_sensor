import json

import numpy as np
import pandas as pd

from src.config import NAB_ENSEMBLE
from src.data import ROOT, load_path, split
from src.detectors import DETECTORS
from src.holdout import frozen, published_alarms
from src.metrics import regions
from src.postprocess import alarms_for, calibrate

RESULTS = ROOT / "results"
DETS = ("autoencoder", "matrix_profile_raw", "ensemble", "zscore")


def fp_runs(alarms: np.ndarray, outside: np.ndarray, gap: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(alarms & outside)
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap)
    starts, ends = np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, ends))


def main():
    series = json.loads((RESULTS / "holdout_series.json").read_text())
    thresholds = json.loads((RESULTS / "holdout_published_thresholds.json").read_text())
    rows = []
    for name, spec in series.items():
        s = load_path(name, *spec)
        tr, va, te = split(len(s.x), 0.4, 0.2)
        idx = np.arange(len(s.x))
        normal = ~s.in_window()
        fit = normal & (idx < tr.stop)
        x_fit = np.where(normal, s.x, np.nan)[tr]
        val = (idx >= va.start) & (idx < va.stop) & normal
        test = idx >= te.start
        t = s.t[test]
        _, in_window, after = regions(t, s.windows, s.context)
        outside = ~in_window & ~after
        med = np.median(s.x[fit])
        mad = np.median(np.abs(s.x[fit] - med)) * 1.4826 or np.std(s.x[fit]) or 1.0
        lo, hi = np.quantile(s.x[fit], [0.001, 0.999])
        xt = s.x[test]

        alarms = {}
        for det in DETS:
            if det == "ensemble":
                w2 = int(round(NAB_ENSEMBLE["w_factor"] * s.w))
                members = [DETECTORS[d](w2, 0).fit(x_fit).score(s.x) for d in NAB_ENSEMBLE["members"]]
                score = np.mean([calibrate(m, m[val]) for m in members], axis=0)
            else:
                score = DETECTORS[det](s.w, 0).fit(x_fit).score(s.x)
            q, span, k = frozen(det)
            alarms[det] = alarms_for(score, val, q, int(round(span * s.w)), int(round(k * s.w)))[test]
        alarms["contextOSE"] = published_alarms("contextOSE", spec[0], s.t, thr=thresholds["contextOSE"])[test]

        for det, a in alarms.items():
            for start, end in fp_runs(a, outside, s.w):
                recent = xt[max(0, start - s.w):start + 1]
                rows.append({
                    "series": name, "category": spec[0].split("/")[0], "detector": det,
                    "position": start / len(xt), "length": end - start + 1, "length_w": (end - start + 1) / s.w,
                    "level_shift_mad": abs(np.median(recent) - med) / mad,
                    "spread_ratio": (np.std(recent) + 1e-9) / (np.std(s.x[fit]) + 1e-9),
                    "outside_train_range": bool((xt[start:end + 1] < lo).any() or (xt[start:end + 1] > hi).any()),
                    "test_level_shift_mad": abs(np.median(xt) - med) / mad,
                })
        print(f"{name}: " + " ".join(f"{d}={sum(r['series'] == name and r['detector'] == d for r in rows)}" for d in alarms), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "fp_analysis.csv", index=False)
    df["kind"] = np.select(
        [df["level_shift_mad"] > 3, df["spread_ratio"] > 2, df["outside_train_range"]],
        ["level shift", "variance change", "spike"], "in-range pattern")
    print("\nfalse-alarm runs per category")
    print(df.pivot_table(index="category", columns="detector", values="length", aggfunc="count", fill_value=0))
    print("\nkind of false alarm (share)")
    print(pd.crosstab(df["kind"], df["detector"], normalize="columns").round(2))
    print("\nposition in test part (share)")
    print(pd.crosstab(pd.cut(df["position"], [0, .25, .5, .75, 1.0], include_lowest=True), df["detector"], normalize="columns").round(2))
    print("\nmedian run length in context lengths")
    print(df.groupby("detector")["length_w"].median().round(2))
    print("\nseries with the most false alarms (autoencoder)")
    print(df[df["detector"] == "autoencoder"].groupby("series").agg(
        fp=("length", "count"), test_shift=("test_level_shift_mad", "first")).sort_values("fp", ascending=False).head(10).round(2))


if __name__ == "__main__":
    main()
