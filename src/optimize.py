import argparse
import itertools
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data import ROOT
from src.metrics import counts, evaluate, prf, regions
from src.multivariate import MV_DETECTORS, build
from src.postprocess import alarms_for, ensemble
from src.run_skab import EXCLUDED, FIT, VAL
from src.skab import FILES, load

RESULTS = ROOT / "results"
QS = (0.95, 0.98, 0.99, 0.995, 0.999, 0.9995)
SPANS = (0.0, 0.25, 0.5)
DEBOUNCE = (0.0, 0.1, 0.25)
DETRENDS = (0, 60, 180)


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


def nab_cases() -> list[Case]:
    scores = pd.read_parquet(RESULTS / "scores.parquet")
    meta = json.loads((RESULTS / "meta.json").read_text())
    cases = []
    for name, m in meta.items():
        df = scores[scores["series"] == name]
        t = pd.DatetimeIndex(df[df["detector"] == df["detector"].iloc[0]]["t"])
        windows = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in m["windows"]]
        normal = np.ones(len(t), dtype=bool)
        for a, b in windows:
            normal &= ~((t >= a) & (t <= b))
        case = Case(
            name, t, (t >= pd.Timestamp(m["val_start"])) & (t < pd.Timestamp(m["test_start"])) & normal,
            t >= pd.Timestamp(m["test_start"]), windows, [pd.Timestamp(p) for p in m["points"]],
            m["w"], pd.Timedelta(m["context"]),
        )
        for det, g in df.groupby("detector", sort=False):
            case.scores[(det, "raw")] = g["score"].to_numpy()
        cases.append(case)
    return cases


def skab_cases(w: int) -> list[Case]:
    cases = []
    for name in [f for f in FILES if f not in EXCLUDED]:
        e = load(name)
        windows = e.windows()
        idx = np.arange(len(e.t))
        case = Case(name, e.t, (idx >= FIT) & (idx < VAL), idx >= VAL, windows, [a for a, _ in windows],
                    w, pd.Timedelta(seconds=w))
        for span in DETRENDS:
            for det in MV_DETECTORS:
                model = build(det, w, 0, span).fit(e.X[:FIT])
                case.scores[(det, f"detrend{span}" if span else "raw")] = model.score(e.X)[0]
        cases.append(case)
        print(f"  scored {name}", flush=True)
    return cases


def add_ensembles(cases: list[Case]) -> None:
    for case in cases:
        for variant in sorted({v for _, v in case.scores}):
            group = {d: s for (d, v), s in case.scores.items() if v == variant and d != "ensemble"}
            case.scores[("ensemble", variant)] = ensemble(group, case.val)


def grid(cases: list[Case]) -> pd.DataFrame:
    rows = []
    for case in cases:
        for (det, variant), score in case.scores.items():
            for q, span, k in itertools.product(QS, SPANS, DEBOUNCE):
                m = run_config(case, score, q, span, k)
                rows.append({"case": case.name, "detector": det, "variant": variant, "q": q, "span": span, "k": k,
                             **{x: m[x] for x in ("events", "tp", "fp", "f1", "delay_h")}})
    return pd.DataFrame(rows)


def leave_one_out(cases: list[Case], g: pd.DataFrame, n_null: int, q0: float) -> pd.DataFrame:
    rows, pool = [], []
    for det in g["detector"].unique():
        entries = {"default": [], "tuned": []}
        variants = list(dict.fromkeys(["raw", *sorted(g.loc[g["detector"] == det, "variant"].unique())]))
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


def summarize(loo: pd.DataFrame) -> pd.DataFrame:
    s = loo.groupby(["detector", "setting"]).agg(
        f1=("f1", "mean"), chance_f1=("chance_f1", "mean"), tp=("tp", "sum"), fp=("fp", "sum"),
        events=("events", "sum"), significant=("p_value", lambda p: (p < 0.05).mean()), delay_h=("delay_h", "median"),
    )
    return s.unstack("setting")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["nab", "skab"], choices=["nab", "skab"])
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--skab-w", type=int, default=20)
    args = parser.parse_args()

    for ds in args.datasets:
        t0 = time.perf_counter()
        cases = nab_cases() if ds == "nab" else skab_cases(args.skab_w)
        add_ensembles(cases)
        g = grid(cases)
        loo, pool = leave_one_out(cases, g, args.n_null, 0.995 if ds == "nab" else 0.99)
        pool.to_csv(RESULTS / f"opt_{ds}_pooled.csv", index=False)
        g.to_csv(RESULTS / f"opt_{ds}_grid.csv", index=False)
        loo.to_csv(RESULTS / f"opt_{ds}_loo.csv", index=False)
        print(f"\n{ds} ({time.perf_counter() - t0:.0f}s)")
        print(summarize(loo).round(2).to_string())
        print(pool.pivot(index="detector", columns="setting").round(3).to_string())
        chosen = loo[loo["setting"] == "tuned"].groupby("detector")[["variant", "q", "span", "k"]].agg(
            lambda c: c.value_counts().index[0])
        print(chosen.to_string())


if __name__ == "__main__":
    main()
