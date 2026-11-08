import argparse
import json
import re
import time

import numpy as np
import pandas as pd

from src.adaptive import refractory
from src.cv import acf_period
from src.data import RAW, ROOT
from src.detectors import DETECTORS
from src.metrics import evaluate
from src.postprocess import calibrate, debounce, smooth

RESULTS = ROOT / "results"
UCR = RAW / "ucr"
FIT_CAP = 20000
TOLERANCE = 100


def series() -> list[dict]:
    out = []
    for f in sorted(UCR.glob("*_UCR_Anomaly_*.txt")):
        m = re.match(r"(\d+)_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt", f.name)
        out.append({"file": f, "id": int(m[1]), "name": m[2], "train_end": int(m[3]), "begin": int(m[4]), "end": int(m[5])})
    return out


def context(x_train: np.ndarray) -> int:
    period, _ = acf_period(x_train[-FIT_CAP:], 8, 2000)
    return int(np.clip(period or 100, 8, 1000))


def score(det: str, w: int, x_fit: np.ndarray, x: np.ndarray, val: np.ndarray, ensemble: dict) -> np.ndarray:
    if det == "ensemble":
        w2 = int(round(ensemble["w_factor"] * w))
        members = [DETECTORS[d](w2, 0).fit(x_fit).score(x) for d in ensemble["members"]]
        return np.mean([calibrate(m, m[val]) for m in members], axis=0)
    return DETECTORS[det](w, 0).fit(x_fit).score(x)


def alarms(sc: np.ndarray, val: np.ndarray, w: int, cfg: dict) -> np.ndarray:
    sm = smooth(sc, int(round(cfg["span"] * w)))
    raw = np.nan_to_num(sm, nan=-np.inf) > np.nanquantile(sm[val], cfg["q"])
    return refractory(debounce(raw, int(round(cfg["k"] * w))), int(cfg["refractory"] * w))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(RESULTS / "final_config.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    final = json.loads(open(args.config).read())
    rows = []
    for spec in series()[: args.limit or None]:
        t0 = time.perf_counter()
        x = np.fromstring(spec["file"].read_text(), sep=" ")
        n, tr = len(x), spec["train_end"]
        w = context(x[:tr])
        fit_end = int(tr * 0.75)
        x_fit = x[max(0, fit_end - FIT_CAP):fit_end]
        idx = np.arange(n)
        val = (idx >= fit_end) & (idx < tr)
        test = idx >= tr
        t = pd.date_range("2000-01-01", periods=n, freq="s")
        windows = [(t[spec["begin"]], t[min(spec["end"], n - 1)])]
        for det, cfg in final["detectors"].items():
            sc = score(det, w, x_fit, x, val, final.get("ensemble", {}))
            top = tr + int(np.nanargmax(np.nan_to_num(sc[test], nan=-np.inf)))
            hit = spec["begin"] - TOLERANCE <= top <= spec["end"] + TOLERANCE
            a = alarms(sc, val, w, cfg)[test]
            m = evaluate(t[test], a, windows, [windows[0][0]], gap=w, grace=pd.Timedelta(seconds=w))
            span = min(spec["end"] + TOLERANCE, n - 1) - max(spec["begin"] - TOLERANCE, tr) + 1
            rows.append({"id": spec["id"], "name": spec["name"], "n": n, "w": w, "detector": det, "top1": hit,
                         "chance_top1": span / (n - tr),
                         **{k: m[k] for k in ("tp", "fp", "f1")}})
        print(f"{spec['id']:3d} {spec['name']:28s} n={n:7d} w={w:4d} "
              + " ".join(f"{r['detector']}={'x' if r['top1'] else '.'}" for r in rows[-len(final['detectors']):])
              + f" {time.perf_counter() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "ucr_metrics.csv", index=False)
    df = pd.DataFrame(rows)
    summary = df.groupby("detector").agg(series=("id", "count"), top1=("top1", "mean"),
                                         chance_top1=("chance_top1", "mean"), tp=("tp", "sum"),
                                         fp=("fp", "sum"), mean_f1=("f1", "mean"))
    summary["pooled_f1"] = 2 * summary["tp"] / (2 * summary["tp"] + summary["fp"] + (summary["series"] - summary["tp"]))
    summary.to_csv(RESULTS / "ucr_summary.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
