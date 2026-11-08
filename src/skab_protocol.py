import argparse
import io
import pickle
import urllib.request

import numpy as np
import pandas as pd

from src.config import SKAB_VARIANT
from src.data import RAW, ROOT
from src.metrics import evaluate
from src.multivariate import MV_DETECTORS, build
from src.postprocess import alarms_for, calibrate
from src.run_skab import FIT, VAL
from src.skab import FILES, fetch

RESULTS = ROOT / "results"
TRAIN = 400
WINDOW = pd.Timedelta(seconds=60)
PROFILES = {"Standard": (1.0, -0.11, -1.0), "LowFP": (1.0, -0.22, -1.0), "LowFN": (1.0, -0.11, -2.0)}
PUBLISHED = {
    "Conv-AE": "Conv_AE", "MSET": "MSET", "T-squared+Q": "T2-q", "LSTM-AE": "LSTM_AE", "T-squared": "T2",
    "Vanilla AE": "Vanilla_AE", "Vanilla LSTM": "Vanilla_LSTM", "Isolation forest": "Isolation_Forest",
    "MSCRED": "MSCRED", "ArimaFD": "Arima_anomaly_detection",
}
LEADERBOARD = {
    "Conv-AE": (0.78, 13.55, 28.02, 23.61), "MSET": (0.78, 39.73, 14.13, 13.84),
    "T-squared+Q": (0.76, 26.62, 24.92, 25.35), "LSTM-AE": (0.74, 29.96, 25.92, 23.51),
    "T-squared": (0.66, 19.21, 42.6, 19.54), "Vanilla AE": (0.39, 2.59, 75.15, 11.41),
    "Vanilla LSTM": (0.54, 12.54, 59.53, 11.31), "Isolation forest": (0.29, 2.56, 82.89, 26.16),
    "MSCRED": (0.36, 49.94, 69.88, 32.42), "ArimaFD": (None, None, None, -0.09),
}
SAFE = {
    ("numpy", "dtype"), ("numpy", "ndarray"), ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("pandas.core.series", "Series"), ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas.core.internals.blocks", "new_block"), ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.datetimes", "DatetimeIndex"),
    ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"), ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.indexes.base", "Index"), ("pandas.core.arrays.datetimes", "DatetimeArray"),
    ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"), ("builtins", "slice"),
    ("pandas.core.internals.blocks", "Block"), ("pandas.core.internals.blocks", "NumpyBlock"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"), ("pandas.core.arrays.datetimes", "_unpickle"),
    ("pandas._libs.tslibs.offsets", "Second"), ("pandas._libs.tslibs.offsets", "_unpickle"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in SAFE:
            raise pickle.UnpicklingError(f"refused {module}.{name}")
        return super().find_class(module, name)


def frame(name: str) -> pd.DataFrame:
    return pd.read_csv(fetch(name), sep=";", index_col="datetime", parse_dates=True)


def binary(trues: list[pd.Series], preds: list[pd.Series]) -> tuple[float, float, float]:
    tp = fp = fn = tn = 0
    for t, p in zip(trues, preds):
        t, p = t.to_numpy() == 1, p.reindex(t.index).fillna(0).to_numpy() == 1
        tp, fp, fn, tn = tp + (t & p).sum(), fp + (~t & p).sum(), fn + (t & ~p).sum(), tn + (~t & ~p).sum()
    return tp / (tp + (fn + fp) / 2), 100 * fp / (fp + tn), 100 * fn / (fn + tp)


def changepoints(pred: pd.Series) -> pd.Series:
    cp = pred.diff().abs()
    cp.iloc[0] = pred.iloc[0]
    return cp


def boundaries(cp: pd.Series) -> list[list[pd.Timestamp]]:
    out = [[t, t + WINDOW] for t in cp.index[cp.to_numpy() == 1]]
    for i in range(len(out) - 1):
        if out[i][1] >= out[i + 1][0]:
            out[i + 1][0] = out[i][1]
    return out


def tanh_scale(a_tp: float, a_fp: float, left, hit, right, n: int = 1000) -> float:
    x = np.linspace(-np.pi / 2, np.pi / 2, n)
    y = (a_tp - a_fp) / 2 * -np.tanh(x) / np.tanh(np.pi / 2) + (a_tp - a_fp) / 2 + a_fp
    return y[min(int((hit - left) / (right - left) * n), n - 1)]


def nab(true_cps: list[pd.Series], pred_cps: list[pd.Series]) -> dict:
    raw = {p: np.zeros(3) for p in PROFILES}
    misses = fps = 0
    for true, pred in zip(true_cps, pred_cps):
        bounds = boundaries(true)
        times = pred.index[pred.fillna(0).to_numpy() == 1]
        inside = np.zeros(len(times), dtype=bool)
        hits = []
        for left, right in bounds:
            m = (times >= left) & (times <= right)
            inside |= m
            hits.append((left, times[m][0], right) if m.any() else None)
        n_fp, n_fn = int((~inside).sum()), sum(h is None for h in hits)
        misses, fps = misses + n_fn, fps + n_fp
        for prof, (a_tp, a_fp, a_fn) in PROFILES.items():
            score = a_fp * n_fp + a_fn * n_fn + sum(tanh_scale(a_tp, a_fp, *h) for h in hits if h)
            raw[prof] += (score, len(bounds) * a_fn, len(bounds) * a_tp)
    out = {f"nab_{p.lower()}": 100 * (s - null) / (perfect - null) for p, (s, null, perfect) in raw.items()}
    return {**out, "missed_cps": misses, "fps": fps}


def leaderboard_metrics(names: list[str], preds: dict[str, pd.Series]) -> dict:
    frames = {n: frame(n).iloc[TRAIN:] for n in names}
    f1, far, mar = binary([frames[n]["anomaly"] for n in names], [preds[n] for n in names])
    scores = nab([frames[n]["changepoint"] for n in names],
                 [changepoints(preds[n].reindex(frames[n].index).fillna(0)) for n in names])
    return {"f1": f1, "far": far, "mar": mar, **scores}


def event_metrics(names: list[str], preds: dict[str, pd.Series]) -> dict:
    tp = fp = ev = 0
    for n in names:
        f = frame(n).sort_index().iloc[TRAIN:]
        y = np.r_[False, f["anomaly"].to_numpy() == 1, False]
        s, e = np.flatnonzero(~y[:-1] & y[1:]), np.flatnonzero(y[:-1] & ~y[1:]) - 1
        windows = [(f.index[a], f.index[b]) for a, b in zip(s, e)]
        alarms = preds[n].reindex(f.index).fillna(0).to_numpy() == 1
        m = evaluate(f.index, alarms, windows, [a for a, _ in windows], gap=20, grace=pd.Timedelta(seconds=20))
        tp, fp, ev = tp + m["tp"], fp + m["fp"], ev + m["events"]
    p, r = tp / max(tp + fp, 1), tp / ev
    return {"event_tp": tp, "event_fp": fp, "event_f1": 2 * p * r / (p + r) if p + r else 0.0}


def published(key: str) -> dict[str, pd.Series]:
    local = RAW / "skab_results" / f"results-{key}.pkl"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/waico/SKAB/master/results/results-{key}.pkl", local)
    series = SafeUnpickler(io.BytesIO(local.read_bytes())).load()
    index = {n: frame(n).index[TRAIN:] for n in FILES}
    out = {}
    for s in series:
        overlap = {n: s.index.isin(idx).mean() for n, idx in index.items() if s.index[0] in idx}
        if overlap and max(overlap.values()) == 1.0:
            out[max(overlap, key=overlap.get)] = s.astype(float)
    return out


def tuned_configs(detector: str) -> dict:
    loo = pd.read_csv(RESULTS / "opt_skab_loo.csv")
    loo = loo[(loo["stage"] == "A_previous") & (loo["setting"] == "tuned") & (loo["detector"] == detector)]
    cfg = {r.case: (r.variant, r.q, r.span, r.k) for r in loo.itertuples()}
    g = pd.read_csv(RESULTS / "opt_skab_grid.csv")
    g = g[(g["detector"] == detector) & (g["variant"].str.split("|").str[0] == SKAB_VARIANT)]
    cfg["_all"] = g.groupby(["variant", "q", "span", "k"])["f1"].mean().idxmax()
    return cfg


def ours(detector: str, setting: str, detrend: int = 60) -> dict[str, pd.Series]:
    cfg = tuned_configs(detector) if setting == "tuned" else {}
    out = {}
    for name in FILES:
        f = frame(name)
        X = f.drop(columns=["anomaly", "changepoint"]).to_numpy(dtype=float)
        idx = np.arange(len(X))
        val = (idx >= FIT) & (idx < VAL)
        variant, q, span, k = cfg.get(name, cfg.get("_all")) if cfg else (SKAB_VARIANT, 0.99, 0.0, 0.0)
        w = int(variant.split("|")[0].split("-w")[1])
        if detector == "ensemble":
            members = variant.split("|")[1].split("+")
            parts = [build(d, w, 0, detrend).fit(X[:FIT]).score(X)[0] for d in members]
            score = np.mean([calibrate(s, s[val]) for s in parts], axis=0)
        else:
            score = build(detector, w, 0, detrend).fit(X[:FIT]).score(X)[0]
        alarms = alarms_for(score, val, q, int(round(span * w)), int(round(k * w)))
        out[name] = pd.Series(alarms.astype(float), index=f.index).iloc[TRAIN:]
        print(f"  {detector} {setting} {name}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", nargs="+", default=[*MV_DETECTORS, "ensemble"])
    args = parser.parse_args()

    always = {n: pd.Series(1.0, index=frame(n).index[TRAIN:]) for n in FILES}
    rows = [{"source": "baseline", "detector": "always alarm", "setting": "-", "files": len(FILES),
             **leaderboard_metrics(FILES, always), **event_metrics(FILES, always)}]
    for label, key in PUBLISHED.items():
        try:
            preds = published(key)
        except Exception as e:
            print(f"skip {label}: {e}")
            continue
        names = [n for n in FILES if n in preds]
        lb = LEADERBOARD[label]
        rows.append({"source": "published", "detector": label, "setting": "as published", "files": len(names),
                     **leaderboard_metrics(names, preds), **event_metrics(names, preds),
                     "reported_f1": lb[0], "reported_nab_standard": lb[3]})
        print(rows[-1], flush=True)
    for det in args.detectors:
        for setting in ("default", "tuned"):
            if det == "ensemble" and setting == "default":
                continue
            preds = ours(det, setting)
            rows.append({"source": "ours", "detector": det, "setting": setting, "files": len(FILES),
                         **leaderboard_metrics(FILES, preds), **event_metrics(FILES, preds)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "skab_leaderboard.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
