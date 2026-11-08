import argparse
import json
import time

import numpy as np
import pandas as pd

from src.data import ROOT, SERIES, load, split
from src.detectors import DETECTORS
from src.metrics import evaluate

RESULTS = ROOT / "results"


def run(names: list[str], detectors: list[str], q: float, train: float, val: float):
    rows, scores, meta = [], [], {}
    for name in names:
        s = load(name)
        tr, va, te = split(len(s.x), train, val)
        normal = ~s.in_window()
        x_fit = np.where(normal, s.x, np.nan)[tr]
        meta[name] = {
            "w": s.w,
            "val_start": str(s.t[va.start]),
            "test_start": str(s.t[te.start]),
            "windows": [[str(a), str(b)] for a, b in s.windows],
        }
        print(f"{name}: n={len(s.x)} w={s.w} events(train/val/test)="
              f"{[sum(((s.t[sl] >= a) & (s.t[sl] <= b)).any() for a, b in s.windows) for sl in (tr, va, te)]}")
        for det in detectors:
            t0 = time.perf_counter()
            score = DETECTORS[det](s.w).fit(x_fit).score(s.x)
            thr = np.nanquantile(score[va][normal[va]], q)
            alarms = np.nan_to_num(score, nan=-np.inf) > thr
            m = evaluate(s.t[te], alarms[te], s.windows, s.points, gap=s.w)
            rows.append({"series": name, "detector": det, "threshold": thr, **m})
            scores.append(pd.DataFrame({
                "series": name, "detector": det, "t": s.t, "x": s.x, "score": score, "alarm": alarms,
            }))
            print(f"  {det:15s} f1={m['f1']:.2f} delay={m['delay_h']:.1f}h  {time.perf_counter() - t0:.1f}s", flush=True)
    return pd.DataFrame(rows), pd.concat(scores, ignore_index=True), meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", nargs="+", default=list(SERIES), choices=list(SERIES))
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
    parser.add_argument("--q", type=float, default=0.995, help="score quantile on normal validation points")
    parser.add_argument("--train", type=float, default=0.4)
    parser.add_argument("--val", type=float, default=0.2)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    metrics, scores, meta = run(args.series, args.detectors, args.q, args.train, args.val)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    scores.to_parquet(RESULTS / "scores.parquet", index=False)
    (RESULTS / "meta.json").write_text(json.dumps(meta, indent=2))
    print(metrics.pivot(index="series", columns="detector", values="f1").round(2))


if __name__ == "__main__":
    main()
