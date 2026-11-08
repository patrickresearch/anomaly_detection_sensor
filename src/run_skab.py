import argparse
import json
import time

import numpy as np
import pandas as pd

from src.data import ROOT
from src.metrics import counts, evaluate, prf, regions
from src.multivariate import MV_DETECTORS
from src.skab import FILES, SENSORS, load

RESULTS = ROOT / "results"
EXCLUDED = {"other/2"}
FIT, VAL = 300, 400


def top_sensor(contrib: np.ndarray) -> tuple[str, np.ndarray]:
    share = contrib / contrib.sum()
    return SENSORS[int(np.argmax(share))], share


def benchmark(detectors, seeds, w, q):
    rows, attribution, scores, meta, pool = [], [], [], {}, {}
    for name in [f for f in FILES if f not in EXCLUDED]:
        e = load(name)
        cause, plausible = e.cause
        te = slice(VAL, len(e.t))
        windows = e.windows()
        grace = pd.Timedelta(seconds=w)
        r = e.y[te].mean()
        meta[name] = {
            "cause": cause, "plausible": sorted(plausible), "point_f1_always": 2 * r / (1 + r),
            "val_start": str(e.t[FIT]), "test_start": str(e.t[VAL]),
            "windows": [[str(a), str(b)] for a, b in windows],
        }
        for det in detectors:
            cls = MV_DETECTORS[det]
            for seed in seeds if cls.stochastic else seeds[:1]:
                model = cls(w, seed).fit(e.X[:FIT])
                score, C = model.score(e.X)
                thr = np.nanquantile(score[FIT:VAL], q)
                alarms = np.nan_to_num(score, nan=-np.inf) > thr
                m = evaluate(e.t[te], alarms[te], windows, [a for a, _ in windows], gap=w, grace=grace)
                m["delay_s"] = m.pop("delay_h") * 3600
                onset = int(np.argmax(e.y))
                rows.append({
                    "file": name, "detector": det, "seed": seed, "threshold": thr,
                    "active_at_onset": bool(alarms[onset - 1]), **m,
                })
                masks, in_window, after = regions(e.t[te], windows, grace)
                pool.setdefault((det, seed), []).append((alarms[te], masks, ~in_window & ~after, len(masks)))
                for a, b, mask in masks:
                    hits = np.flatnonzero(mask & alarms[te])
                    if len(hits):
                        top, share = top_sensor(C[te][hits].mean(0))
                        attribution.append({
                            "file": name, "detector": det, "seed": seed, "cause": cause, "top": top,
                            "plausible": top in plausible, **dict(zip(SENSORS, share)),
                        })
                if seed == seeds[0]:
                    df = pd.DataFrame(C.astype(np.float32), columns=[f"c_{s}" for s in SENSORS])
                    df = pd.concat([pd.DataFrame(e.X.astype(np.float32), columns=SENSORS), df], axis=1)
                    df.insert(0, "file", name)
                    df.insert(1, "detector", det)
                    df.insert(2, "t", e.t)
                    df.insert(3, "label", e.y)
                    df.insert(4, "score", score)
                    df.insert(5, "alarm", alarms)
                    scores.append(df)
        print(f"{name:10s} {cause:34s} " + "  ".join(
            f"{r['detector']}={r['f1']:.2f}" for r in rows if r["file"] == name and r["seed"] == seeds[0]), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(attribution), pd.concat(scores, ignore_index=True), meta, pool


def pooled(pool: dict, gap: int, n_null: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for (det, s), files in pool.items():
        def total(shift: bool) -> float:
            tp = fp = ev = 0
            for alarms, masks, outside, events in files:
                a = np.roll(alarms, rng.integers(1, len(alarms))) if shift else alarms
                t, f = counts(a, masks, outside, gap)
                tp, fp, ev = tp + t, fp + f, ev + events
            return tp, fp, ev
        tp, fp, ev = total(False)
        precision, recall, f1 = prf(tp, fp, ev)
        null = np.array([prf(*total(True))[2] for _ in range(n_null)])
        out.append({
            "detector": det, "seed": s, "events": ev, "tp": tp, "fp": fp,
            "precision": precision, "recall": recall, "f1": f1,
            "chance_f1": null.mean(), "p_value": (1 + (null >= f1).sum()) / (1 + n_null),
        })
    return pd.DataFrame(out)


def injection(detectors, w, q, length=200, mags=(3.0, 6.0), starts=10):
    X = load("anomaly-free/anomaly-free").X
    positions = np.linspace(VAL, len(X) - length - 1, starts).astype(int)
    rows = []
    for det in detectors:
        for p in positions:
            model = MV_DETECTORS[det](w, 0).fit(X[p - VAL: p - VAL + FIT])
            thr = np.nanquantile(model.score(X[p - VAL: p])[0], q)
            clean = bool((model.score(X[p - w: p + length])[0][w:] > thr).any())
            for j, sensor in enumerate(SENSORS):
                for kind in ("step", "drift"):
                    ramp = np.ones(length) if kind == "step" else np.linspace(0, 1, length)
                    for mag in mags:
                        Xi = X[p - w: p + length].copy()
                        Xi[w:, j] += mag * model.sd[j] * ramp
                        score, C = model.score(Xi)
                        hits = np.flatnonzero(score[w:] > thr)
                        top = SENSORS[int(np.argmax(C[w + hits].mean(0)))] if len(hits) else None
                        rows.append({
                            "detector": det, "sensor": sensor, "kind": kind, "mag": mag, "start": int(p),
                            "alarm_without_fault": clean, "detected": bool(len(hits)), "top": top,
                            "correct": top == sensor, "delay_s": int(hits[0]) if len(hits) else np.nan,
                        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", nargs="+", default=list(MV_DETECTORS), choices=list(MV_DETECTORS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--w", type=int, default=20, help="trailing window in points (1 s sampling)")
    parser.add_argument("--q", type=float, default=0.99, help="score quantile on validation points 300-400")
    parser.add_argument("--n-null", type=int, default=1000)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    rows, attribution, scores, meta, pool = benchmark(args.detectors, args.seeds, args.w, args.q)
    summary = pooled(pool, args.w, args.n_null)
    inj = injection(args.detectors, args.w, args.q)
    rows.to_csv(RESULTS / "skab_metrics.csv", index=False)
    summary.to_csv(RESULTS / "skab_pooled.csv", index=False)
    attribution.to_csv(RESULTS / "skab_attribution.csv", index=False)
    inj.to_csv(RESULTS / "skab_injection.csv", index=False)
    scores.to_parquet(RESULTS / "skab_scores.parquet", index=False)
    (RESULTS / "skab_meta.json").write_text(json.dumps({"w": args.w, "q": args.q, "files": meta}, indent=2))

    print(summary.groupby("detector")[["f1", "chance_f1", "p_value", "fp"]].mean().round(3).to_string())
    print(rows.groupby("detector")[["point_f1", "delay_s"]].median().round(2).to_string())
    print(attribution.groupby("detector")["plausible"].mean().round(2).to_string())
    det = inj[inj["detected"]]
    print(inj.groupby(["detector", "mag"])[["alarm_without_fault", "detected"]].mean().join(
        det.groupby(["detector", "mag"])["correct"].mean()).round(2).to_string())
    print(f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
