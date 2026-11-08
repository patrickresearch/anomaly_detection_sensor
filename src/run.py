import argparse
import json
import time

import numpy as np
import pandas as pd

from src.data import ROOT, SERIES, load, remove_weekly, split
from src.detectors import DETECTORS
from src.metrics import evaluate

RESULTS = ROOT / "results"
QS = (0.95, 0.98, 0.99, 0.995, 0.999, 0.9995)


def run(names: list[str], detectors: list[str], q: float, train: float, val: float, seeds: list[int], n_null: int):
    rows, sweep, scores, meta = [], [], [], {}
    for name in names:
        s = load(name)
        tr, va, te = split(len(s.x), train, val)
        normal = ~s.in_window()
        fit = normal & (np.arange(len(s.x)) < tr.stop)
        x = remove_weekly(s.t, s.x, fit) if s.weekly else s.x
        x_fit = np.where(normal, x, np.nan)[tr]
        meta[name] = {
            "w": s.w,
            "weekly": s.weekly,
            "val_start": str(s.t[va.start]),
            "test_start": str(s.t[te.start]),
            "context": str(s.context),
            "windows": [[str(a), str(b)] for a, b in s.windows],
            "points": [str(p) for p in s.points],
        }
        print(f"{name}: n={len(s.x)} w={s.w} weekly={s.weekly} events(train/val/test)="
              f"{[int(sum(((s.t[sl] >= a) & (s.t[sl] <= b)).any() for a, b in s.windows)) for sl in (tr, va, te)]}")
        ev = dict(windows=s.windows, points=s.points, gap=s.w, grace=s.context)
        for det in detectors:
            cls = DETECTORS[det]
            for seed in seeds if cls.stochastic else seeds[:1]:
                t0 = time.perf_counter()
                score = cls(s.w, seed).fit(x_fit).score(x)
                ref = score[va][normal[va]]
                filled = np.nan_to_num(score, nan=-np.inf)
                thr = np.nanquantile(ref, q)
                alarms = filled > thr
                m = evaluate(s.t[te], alarms[te], **ev, n_null=n_null)
                key = {"series": name, "detector": det, "seed": seed}
                rows.append({**key, "threshold": thr, **m})
                for qq in QS:
                    mq = evaluate(s.t[te], filled[te] > np.nanquantile(ref, qq), **ev)
                    sweep.append({**key, "q": qq, **{k: mq[k] for k in ("tp", "fp", "precision", "recall", "f1")}})
                if seed == seeds[0]:
                    scores.append(pd.DataFrame({
                        "series": name, "detector": det, "t": s.t, "x": s.x, "score": score, "alarm": alarms,
                    }))
                print(f"  {det:18s} seed={seed} f1={m['f1']:.2f} chance={m['chance_f1']:.2f} p={m['p_value']:.3f} "
                      f"tp={m['tp']}/{m['events']} fp={m['fp']}  {time.perf_counter() - t0:.1f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(sweep), pd.concat(scores, ignore_index=True), meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", nargs="+", default=list(SERIES), choices=list(SERIES))
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
    parser.add_argument("--q", type=float, default=0.995, help="score quantile on normal validation points")
    parser.add_argument("--train", type=float, default=0.4)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-null", type=int, default=1000, help="random circular shifts for the chance baseline")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    metrics, sweep, scores, meta = run(
        args.series, args.detectors, args.q, args.train, args.val, args.seeds, args.n_null
    )
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    sweep.to_csv(RESULTS / "sweep.csv", index=False)
    scores.to_parquet(RESULTS / "scores.parquet", index=False)
    (RESULTS / "meta.json").write_text(json.dumps(meta, indent=2))
    summary = metrics.groupby(["series", "detector"])[["f1", "chance_f1"]].mean().unstack("detector")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
