import argparse
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data import ROOT, SERIES, remove_weekly, split
from src.data import load as load_series
from src.detectors import DETECTORS
from src.metrics import counts, evaluate, prf, regions
from src.multivariate import MV_DETECTORS, build
from src.postprocess import alarms_for, calibrate, debounce, smooth
from src.run_skab import EXCLUDED, FIT, VAL
from src.skab import FILES, load

RESULTS = ROOT / "results"
QS = (0.95, 0.98, 0.99, 0.995, 0.999, 0.9995)
SPANS = (0.0, 0.25, 0.5)
DEBOUNCE = (0.0, 0.1, 0.25)
NAB_W = (0.5, 1.0, 2.0)
SKAB_W = (10, 20, 40)
SKAB_DETREND = 60
NAB_BASE, SKAB_BASE = "w1", "local-w20"


@dataclass
class Case:
    name: str
    t: pd.DatetimeIndex
    val: np.ndarray
    test: np.ndarray
    windows: list
    points: list
    w: int
    grace: pd.Timedelta
    scores: dict = field(default_factory=dict)


def configs(variants, q0: float) -> list[tuple]:
    default = [(variants[0], q0, 0.0, 0.0)]
    rest = [c for c in itertools.product(variants, QS, SPANS, DEBOUNCE) if c != default[0]]
    return default + rest


def run_config(case: Case, score: np.ndarray, q: float, span: float, k: float, n_null: int = 0, entry=None) -> dict:
    alarms = alarms_for(score, case.val, q, int(round(span * case.w)), int(round(k * case.w)))[case.test]
    t = case.t[case.test]
    if entry is not None:
        masks, in_window, after = regions(t, case.windows, case.grace)
        entry.append((alarms, masks, ~in_window & ~after, len(masks), case.w))
    return evaluate(t, alarms, case.windows, case.points, gap=case.w, grace=case.grace, n_null=n_null)


def pooled(entries: list, n_null: int, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    def total(shift: bool) -> float:
        tp = fp = ev = 0
        for alarms, masks, outside, events, gap in entries:
            a = np.roll(alarms, rng.integers(1, len(alarms))) if shift else alarms
            t, f = counts(a, masks, outside, gap)
            tp, fp, ev = tp + t, fp + f, ev + events
        return prf(tp, fp, ev)[2]

    f1 = total(False)
    null = np.array([total(True) for _ in range(n_null)])
    return f1, float(null.mean()), float((1 + (null >= f1).sum()) / (1 + n_null))


def nab_cases(factors=NAB_W) -> list[Case]:
    cases = []
    for name in SERIES:
        s = load_series(name)
        tr, va, te = split(len(s.x), 0.4, 0.2)
        idx = np.arange(len(s.x))
        normal = ~s.in_window()
        x = remove_weekly(s.t, s.x, normal & (idx < tr.stop)) if s.weekly else s.x
        x_fit = np.where(normal, x, np.nan)[tr]
        case = Case(name, s.t, (idx >= va.start) & (idx < va.stop) & normal, idx >= te.start,
                    s.windows, s.points, s.w, s.context)
        for f in factors:
            w = max(4, int(round(f * s.w)))
            for det, cls in DETECTORS.items():
                case.scores[(det, f"w{f:g}")] = cls(w, 0).fit(x_fit).score(x)
        cases.append(case)
        print(f"  scored {name}", flush=True)
    return cases


def skab_cases(ws=SKAB_W, trains=("local", "free")) -> list[Case]:
    free = load("anomaly-free/anomaly-free").X
    shared = {(det, w): build(det, w, 0, SKAB_DETREND).fit(free) for det in MV_DETECTORS for w in ws} if "free" in trains else {}
    cases = []
    for name in [f for f in FILES if f not in EXCLUDED]:
        e = load(name)
        windows = e.windows()
        idx = np.arange(len(e.t))
        case = Case(name, e.t, (idx >= FIT) & (idx < VAL), idx >= VAL, windows, [a for a, _ in windows],
                    20, pd.Timedelta(seconds=20))
        for det in MV_DETECTORS:
            for w in ws:
                for train in trains:
                    model = shared[(det, w)] if train == "free" else build(det, w, 0, SKAB_DETREND).fit(e.X[:FIT])
                    case.scores[(det, f"{train}-w{w}")] = model.score(e.X)[0]
        cases.append(case)
        print(f"  scored {name}", flush=True)
    return cases


def add_ensembles(cases: list[Case]) -> None:
    for case in cases:
        singles = dict(case.scores)
        for variant in sorted({v for _, v in singles}):
            cal = {d: calibrate(sc, sc[case.val]) for (d, v), sc in singles.items() if v == variant}
            dets = sorted(cal)
            for r in range(2, len(dets) + 1):
                for sub in itertools.combinations(dets, r):
                    case.scores[("ensemble", f"{variant}|{'+'.join(sub)}")] = np.mean([cal[d] for d in sub], axis=0)


def grid(cases: list[Case]) -> pd.DataFrame:
    rows = []
    for case in cases:
        t = case.t[case.test]
        for (det, variant), score in case.scores.items():
            for span in SPANS:
                sm = smooth(score, int(round(span * case.w)))
                filled = np.nan_to_num(sm, nan=-np.inf)
                for q in QS:
                    raw = filled > np.nanquantile(sm[case.val], q)
                    for k in DEBOUNCE:
                        m = evaluate(t, debounce(raw, int(round(k * case.w)))[case.test], case.windows, case.points,
                                     gap=case.w, grace=case.grace)
                        rows.append({"case": case.name, "detector": det, "variant": variant, "q": q, "span": span,
                                     "k": k, **{x: m[x] for x in ("events", "tp", "fp", "f1", "delay_h")}})
    return pd.DataFrame(rows)


def base_variant(det: str, variants, base: str) -> str:
    if det != "ensemble":
        return base
    return max((v for v in variants if v.startswith(base + "|")), key=lambda v: v.count("+"))


def leave_one_out(cases: list[Case], g: pd.DataFrame, n_null: int, q0: float, base: str):
    rows, pool = [], []
    for det in g["detector"].unique():
        entries = {"default": [], "tuned": []}
        found = sorted(g.loc[g["detector"] == det, "variant"].unique())
        variants = list(dict.fromkeys([base_variant(det, found, base), *found]))
        cfgs = configs(variants, q0)
        mean_f1 = g[g["detector"] == det].pivot_table(index=["variant", "q", "span", "k"], columns="case", values="f1")
        for case in cases:
            others = mean_f1.drop(columns=case.name).mean(axis=1)
            best = max(cfgs, key=lambda c: (round(others.get(c, -1), 9), -cfgs.index(c)))
            for label, cfg in (("default", cfgs[0]), ("tuned", best)):
                variant, q, span, k = cfg
                m = run_config(case, case.scores[(det, variant)], q, span, k, n_null=n_null, entry=entries[label])
                rows.append({"case": case.name, "detector": det, "setting": label, "variant": variant,
                             "q": q, "span": span, "k": k, **m})
        for label, e in entries.items():
            f1, chance, p = pooled(e, n_null)
            pool.append({"detector": det, "setting": label, "pooled_f1": f1, "pooled_chance_f1": chance, "pooled_p": p})
    return pd.DataFrame(rows), pd.DataFrame(pool)


def auto(g: pd.DataFrame, stages: dict, n_boot: int = 5000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-out where the detector is part of the configuration, compared stage by stage."""
    key = ["detector", "variant", "q", "span", "k"]
    rng = np.random.default_rng(seed)
    per_case, summary, first = [], [], None
    for stage, keep in stages.items():
        sub = g[[keep(d, v) for d, v in zip(g["detector"], g["variant"])]]
        piv = sub.pivot_table(index=key, columns="case", values="f1")
        idx = sub.set_index(["case", *key])
        rows = []
        for case in piv.columns:
            cfg = piv.drop(columns=case).mean(axis=1).idxmax()
            r = idx.loc[(case, *cfg)]
            rows.append({"stage": stage, "case": case, **dict(zip(key, cfg)), **r[["f1", "tp", "fp", "events"]].to_dict()})
        r = pd.DataFrame(rows)
        first = r if first is None else first
        d = r["f1"].to_numpy() - first["f1"].to_numpy()
        boot = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
        summary.append({
            "stage": stage, "mean_f1": r["f1"].mean(),
            "pooled_f1": prf(r["tp"].sum(), r["fp"].sum(), r["events"].sum())[2],
            "tp": r["tp"].sum(), "fp": r["fp"].sum(), "diff_vs_first": d.mean(),
            "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975),
            "most_chosen": (r["detector"] + " " + r["variant"]).value_counts().index[0],
        })
        per_case.append(r)
    return pd.concat(per_case), pd.DataFrame(summary)


STAGES = {
    "nab": {
        "A_previous": lambda d, v: v in ("w1", "w1|autoencoder+iforest+matrix_profile+matrix_profile_raw+zscore"),
        "B_ensemble_subsets": lambda d, v: v.split("|")[0] == "w1",
        "C_context_length": lambda d, v: True,
    },
    "skab": {
        "A_previous": lambda d, v: v == "local-w20" or v == "local-w20|autoencoder+pca+zscore",
        "B_ensemble_subsets": lambda d, v: v.split("|")[0] == "local-w20",
        "C_context_length": lambda d, v: v.startswith("local-"),
        "D_anomaly_free_training": lambda d, v: True,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["nab", "skab"], choices=["nab", "skab"])
    parser.add_argument("--n-null", type=int, default=1000)
    args = parser.parse_args()

    for ds in args.datasets:
        t0 = time.perf_counter()
        cases = nab_cases() if ds == "nab" else skab_cases()
        add_ensembles(cases)
        g = grid(cases)
        g.to_csv(RESULTS / f"opt_{ds}_grid.csv", index=False)
        q0, base = (0.995, NAB_BASE) if ds == "nab" else (0.99, SKAB_BASE)
        ablation, loos, pools = [], [], []
        stages = STAGES[ds]
        for stage, keep in stages.items():
            sub = g[[keep(d, v) for d, v in zip(g["detector"], g["variant"])]]
            loo, pool = leave_one_out(cases, sub, args.n_null, q0, base)
            loos.append(loo.assign(stage=stage))
            pools.append(pool.assign(stage=stage))
            tuned = loo[loo["setting"] == "tuned"].groupby("detector").agg(
                mean_f1=("f1", "mean"), tp=("tp", "sum"), fp=("fp", "sum"), events=("events", "sum"))
            tuned = tuned.join(pool[pool["setting"] == "tuned"].set_index("detector").drop(columns="setting"))
            ablation.append(tuned.assign(stage=stage).reset_index())
            print(f"\n{ds} {stage}\n{tuned.round(3).to_string()}", flush=True)
        pd.concat(loos).to_csv(RESULTS / f"opt_{ds}_loo.csv", index=False)
        pd.concat(pools).to_csv(RESULTS / f"opt_{ds}_pooled.csv", index=False)
        pd.concat(ablation).to_csv(RESULTS / f"opt_{ds}_ablation.csv", index=False)
        per_case, summary = auto(g, stages)
        per_case.to_csv(RESULTS / f"opt_{ds}_auto.csv", index=False)
        summary.to_csv(RESULTS / f"opt_{ds}_auto_summary.csv", index=False)
        print(summary.round(3).to_string())
        print(f"{ds}: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
