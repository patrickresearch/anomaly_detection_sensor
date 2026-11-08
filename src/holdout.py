import argparse
import json

import numpy as np
import pandas as pd

from src.config import NAB_ENSEMBLE, nab_variant
from src.data import ROOT, SERIES, fetch, load_path, remove_weekly, split
from src.detectors import DETECTORS
from src.metrics import evaluate, regions
from src.nab_score import PROFILES, normalized, raw_score
from src.optimize import pooled
from src.postprocess import alarms_for, calibrate

RESULTS = ROOT / "results"
PUBLISHED = ("ARTime", "numenta", "contextOSE", "htmjava", "knncad", "relativeEntropy", "earthgeckoSkyline",
             "randomCutForest", "twitterADVec", "windowedGaussian", "skyline", "random")


def context_rule(t: pd.Series) -> str:
    dt = t.diff().median()
    return str(max(pd.Timedelta("2h"), 12 * dt))


def holdout_series() -> dict[str, tuple[str, str, bool]]:
    windows = json.loads(fetch("labels/combined_windows.json").read_text())
    used = {p for p, _, _ in SERIES.values()}
    out = {}
    for path, w in windows.items():
        if not path.startswith("real") or path in used:
            continue
        t = pd.read_csv(fetch(f"data/{path}"), parse_dates=["timestamp"])["timestamp"]
        if any(pd.Timestamp(b) >= t.iloc[int(len(t) * 0.6)] for _, b in w):
            out[path.split("/")[1].removesuffix(".csv")] = (path, context_rule(t), False)
    return out


def frozen(detector: str) -> tuple[float, float, float]:
    g = pd.read_csv(RESULTS / "opt_nab_grid.csv")
    g = g[(g["detector"] == detector) & (g["variant"] == nab_variant(detector))]
    return g.groupby(["q", "span", "k"])["f1"].mean().idxmax()


def onsets(alarms: np.ndarray) -> np.ndarray:
    return alarms & ~np.r_[False, alarms[:-1]]


def published_alarms(detector: str, path: str, t: pd.DatetimeIndex, profile: str = "standard",
                     thr: float | None = None) -> np.ndarray:
    if thr is None:
        thr = json.loads(fetch("labels/thresholds.json").read_text())[detector][profile]["threshold"]
    return published_score(detector, path, t) >= thr


def published_score(detector: str, path: str, t: pd.DatetimeIndex) -> np.ndarray:
    cat, f = path.split("/")
    df = pd.read_csv(fetch(f"results/{detector}/{cat}/{detector}_{f}"), parse_dates=["timestamp"])
    if len(df) == len(t) and (pd.DatetimeIndex(df["timestamp"]) == t).all():
        score = df["anomaly_score"].to_numpy()
    else:
        score = df.drop_duplicates("timestamp").set_index("timestamp")["anomaly_score"].reindex(t).fillna(0).to_numpy()
    return score


def published_thresholds(series: dict) -> dict[str, float]:
    data = []
    for name, spec in series.items():
        s = load_path(name, *spec)
        test = np.arange(len(s.t)) >= split(len(s.t), 0.4, 0.2)[2].start
        data.append((s, test))
    out = {}
    for det in PUBLISHED:
        scores = [(published_score(det, spec[0], s.t)[test], s, test) for (s, test), spec in zip(data, series.values())]
        pooled_scores = np.concatenate([sc for sc, _, _ in scores])
        official = json.loads(fetch("labels/thresholds.json").read_text())[det]["standard"]["threshold"]
        tail = pooled_scores[pooled_scores >= np.quantile(pooled_scores, 0.9)]
        grid = np.unique(np.r_[np.quantile(tail, np.linspace(0, 1, 300)), official])
        grid = grid[(grid > pooled_scores.min()) & ((pooled_scores >= grid[:, None]).mean(axis=1) <= 0.1)]

        def mean_f1(thr):
            return np.mean([evaluate(s.t[test], sc >= thr, s.windows, s.points, gap=s.w, grace=s.context)["f1"]
                            for sc, s, test in scores])

        out[det] = float(max(grid, key=mean_f1))
    return out


def run(groups: dict, n_null: int):
    rows, entries, nab_parts = [], {}, {}
    thresholds = published_thresholds(groups["tuning"])
    (RESULTS / "holdout_published_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    for group, series in groups.items():
        for name, spec in series.items():
            s = load_path(name, *spec)
            tr, va, te = split(len(s.x), 0.4, 0.2)
            idx = np.arange(len(s.x))
            normal = ~s.in_window()
            x = remove_weekly(s.t, s.x, normal & (idx < tr.stop)) if s.weekly else s.x
            x_fit = np.where(normal, x, np.nan)[tr]
            val = (idx >= va.start) & (idx < va.stop) & normal
            test = idx >= te.start
            t = s.t[test]
            masks, in_window, after = regions(t, s.windows, s.context)
            outside = ~in_window & ~after

            def score(det):
                if det != "ensemble":
                    return DETECTORS[det](s.w, 0).fit(x_fit).score(x)
                w2 = int(round(NAB_ENSEMBLE["w_factor"] * s.w))
                members = [DETECTORS[d](w2, 0).fit(x_fit).score(x) for d in NAB_ENSEMBLE["members"]]
                return np.mean([calibrate(m, m[val]) for m in members], axis=0)

            candidates = {}
            for det in [*DETECTORS, "ensemble"]:
                sc = score(det)
                q, span, k = frozen(det)
                candidates[(det, "frozen")] = alarms_for(sc, val, q, int(round(span * s.w)), int(round(k * s.w)))[test]
                candidates[(det, "default")] = alarms_for(sc, val, 0.995, 0, 0)[test]
            for det in PUBLISHED:
                candidates[(det, "published")] = published_alarms(det, spec[0], s.t)[test]
                candidates[(det, "frozen")] = published_alarms(det, spec[0], s.t, thr=thresholds[det])[test]

            for (det, setting), alarms in candidates.items():
                m = evaluate(t, alarms, s.windows, s.points, gap=s.w, grace=s.context, n_null=n_null)
                key = (group, det, setting)
                entries.setdefault(key, []).append((alarms, masks, outside, len(masks), s.w))
                for prof in PROFILES:
                    a = alarms
                    if det in PUBLISHED and setting == "published" and prof != "standard":
                        a = published_alarms(det, spec[0], s.t, prof)[test]
                    nab_parts.setdefault((*key, prof, "points"), []).append(raw_score(t, a, s.windows, prof))
                    nab_parts.setdefault((*key, prof, "onsets"), []).append(raw_score(t, onsets(a), s.windows, prof))
                rows.append({"group": group, "series": name, "detector": det, "setting": setting,
                             "context": str(s.context), **m})
            ens = next(r["f1"] for r in rows[-len(candidates):] if r["detector"] == "ensemble" and r["setting"] == "frozen")
            print(f"{group} {name}: w={s.w} events={len(masks)} ensemble f1={ens:.2f}", flush=True)

    summary = []
    for (group, det, setting), e in entries.items():
        f1, chance, p = pooled(e, n_null)
        sub = [r for r in rows if (r["group"], r["detector"], r["setting"]) == (group, det, setting)]
        rec = {"group": group, "detector": det, "setting": setting, "series": len(sub),
               "events": sum(r["events"] for r in sub), "tp": sum(r["tp"] for r in sub),
               "fp": sum(r["fp"] for r in sub), "mean_f1": np.mean([r["f1"] for r in sub]),
               "pooled_f1": f1, "pooled_chance_f1": chance, "pooled_p": p}
        for prof in PROFILES:
            for mode in ("points", "onsets"):
                rec[f"nab_{prof}_{mode}"] = normalized(nab_parts[(group, det, setting, prof, mode)], prof)
        summary.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-null", type=int, default=1000)
    args = parser.parse_args()
    groups = {"tuning": dict(SERIES), "holdout": holdout_series()}
    (RESULTS / "holdout_series.json").write_text(json.dumps(groups["holdout"], indent=2))
    rows, summary = run(groups, args.n_null)
    rows.to_csv(RESULTS / "holdout_metrics.csv", index=False)
    summary.to_csv(RESULTS / "holdout_summary.csv", index=False)
    cols = ["group", "detector", "setting", "tp", "fp", "events", "pooled_f1", "pooled_chance_f1", "pooled_p",
            "nab_standard_points", "nab_standard_onsets"]
    print(summary[cols].sort_values(["group", "pooled_f1"], ascending=[True, False]).round(3).to_string())


if __name__ == "__main__":
    main()
