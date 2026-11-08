import argparse
import itertools
import json
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.adaptive import refractory, robust_z
from src.config import NAB_ENSEMBLE
from src.data import RAW, ROOT, SERIES, load_path, remove_weekly, split
from src.detectors import DETECTORS
from src.metrics import evaluate, prf, regions
from src.optimize import pooled
from src.postprocess import calibrate, debounce, smooth

RESULTS = ROOT / "results"
CACHE = RAW.parent / "cache"
DETS = ("zscore", "iforest", "matrix_profile_raw", "autoencoder", "relative_entropy", "ensemble")
QS = (0.95, 0.98, 0.99, 0.995, 0.999, 0.9995)
SPANS = (0.0, 0.25, 0.5)
DEBOUNCE = (0.0, 0.1, 0.25)
REFRACTORY = (0, 2, 5, 10)
HISTORY = (20, 100)
AVG = (0.0, 0.25)
CUTS = (3, 4, 6, 8, 12, 20)
DIVERSE = {"mp_ae_re": ("matrix_profile_raw", "autoencoder", "relative_entropy"),
           "all5": ("zscore", "iforest", "matrix_profile_raw", "autoencoder", "relative_entropy")}


@dataclass
class Case:
    name: str
    category: str
    t: pd.DatetimeIndex
    val: np.ndarray
    test: np.ndarray
    start: int
    windows: list
    points: list
    w: int
    grace: pd.Timedelta
    scores: dict = field(default_factory=dict)


def all_series() -> dict:
    holdout = json.loads((RESULTS / "holdout_series.json").read_text())
    return {**SERIES, **{k: tuple(v) for k, v in holdout.items()}}


def acf_period(x: np.ndarray, lo: int, hi: int, min_acf: float = 0.3) -> tuple[int | None, np.ndarray]:
    v = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    v = v - v.mean()
    f = np.fft.rfft(v, 2 * len(v))
    acf = np.fft.irfft(f * np.conj(f))[:len(v)]
    acf = acf / acf[0] if acf[0] > 0 else acf
    hi = min(hi, len(v) // 3)
    peaks = [lag for lag in range(max(lo, 2), hi) if acf[lag - 1] < acf[lag] >= acf[lag + 1] and acf[lag] > min_acf]
    return (max(peaks, key=lambda lag: acf[lag]) if peaks else None), acf


def preprocess(variant: str, s, normal: np.ndarray, tr: slice) -> tuple[np.ndarray, int]:
    idx = np.arange(len(s.x))
    fit = normal & (idx < tr.stop)
    if variant == "base":
        return (remove_weekly(s.t, s.x, fit) if s.weekly else s.x), s.w
    if variant == "log":
        x = remove_weekly(s.t, s.x, fit) if s.weekly else s.x
        ref = s.x[fit]
        if ref.min() >= 0 and pd.Series(ref).skew() > 1:
            x = np.log1p(np.maximum(s.x, 0))
            x = remove_weekly(s.t, x, fit) if s.weekly else x
        return x, s.w
    if variant == "acf":
        dt = s.t.to_series().diff().median()
        week = int(pd.Timedelta("7D") / dt)
        rule = max(12, int(pd.Timedelta("2h") / dt))
        period, acf = acf_period(np.where(fit, s.x, np.nan)[tr], 4, int(pd.Timedelta("2D") / dt) + 2)
        weekly = week < tr.stop // 2 and acf[week] >= 0.5
        x = remove_weekly(s.t, s.x, fit) if weekly else s.x
        return x, int(np.clip(period or rule, 8, tr.stop // 8))
    raise ValueError(variant)


def build_case(name: str, spec, variants=("base",)) -> Case:
    s = load_path(name, *spec)
    tr, va, te = split(len(s.x), 0.4, 0.2)
    idx = np.arange(len(s.x))
    normal = ~s.in_window()
    val = (idx >= va.start) & (idx < va.stop) & normal
    case = Case(name, spec[0].split("/")[0], s.t, val, idx >= te.start, va.start, s.windows, s.points, s.w, s.context)
    path = CACHE / f"cv_{name}.pkl"
    if path.exists():
        case.scores = pickle.loads(path.read_bytes())
    for variant in variants:
        prefix = "" if variant == "base" else f"{variant}:"
        todo = [d for d in DETS if prefix + d not in case.scores]
        if not todo:
            continue
        x, w = preprocess(variant, s, normal, tr)
        x_fit = np.where(normal, x, np.nan)[tr]
        for det in todo:
            if det == "ensemble":
                w2 = int(round(NAB_ENSEMBLE["w_factor"] * w))
                members = [DETECTORS[d](w2, 0).fit(x_fit).score(x) for d in NAB_ENSEMBLE["members"]]
                case.scores[prefix + det] = np.mean([calibrate(m, m[val]) for m in members], axis=0)
            else:
                case.scores[prefix + det] = DETECTORS[det](w, 0).fit(x_fit).score(x)
        if prefix:
            case.scores[f"{prefix}w"] = w
    for label, members in DIVERSE.items():
        if f"ens:{label}" not in case.scores:
            case.scores[f"ens:{label}"] = np.mean([calibrate(case.scores[m], case.scores[m][val]) for m in members], axis=0)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(case.scores))
    return case


def configs() -> list[tuple]:
    quant = [("quantile", q, sp, k, r) for q, sp, k, r in itertools.product(QS, SPANS, DEBOUNCE, REFRACTORY)]
    like = [("likelihood", h, a, c, r) for h, a, c, r in itertools.product(HISTORY, AVG, CUTS, REFRACTORY)]
    return quant + like


def alarms(case: Case, det: str, cfg: tuple, memo: dict | None = None) -> np.ndarray:
    family, a, b, c, r = cfg
    score = case.scores[det]
    if family == "quantile":
        key = (det, "smooth", b)
        sm = memo.get(key) if memo is not None else None
        if sm is None:
            sm = smooth(score, int(round(b * case.w)))
            if memo is not None:
                memo[key] = sm
        raw = np.nan_to_num(sm, nan=-np.inf) > np.nanquantile(sm[case.val], a)
        out = debounce(raw, int(round(c * case.w)))
    else:
        key = (det, "z", a, b)
        z = memo.get(key) if memo is not None else None
        if z is None:
            z = robust_z(score, case.start, int(a * case.w), max(1, int(round(b * case.w))))
            if memo is not None:
                memo[key] = z
        out = np.nan_to_num(z, nan=-np.inf) > c
    return refractory(out[case.test], r * case.w)


def grid(cases: list[Case], skip: set = frozenset()) -> pd.DataFrame:
    rows, cfgs = [], configs()
    for case in cases:
        t0 = time.perf_counter()
        t = case.t[case.test]
        memo = {}
        for det in [k for k in case.scores if not k.endswith(":w") and k not in skip]:
            for cfg in cfgs:
                m = evaluate(t, alarms(case, det, cfg, memo), case.windows, case.points, gap=case.w, grace=case.grace)
                rows.append({"case": case.name, "category": case.category, "detector": det, "family": cfg[0],
                             "p1": cfg[1], "p2": cfg[2], "p3": cfg[3], "refractory": cfg[4],
                             **{x: m[x] for x in ("events", "tp", "fp", "f1")}})
        print(f"  grid {case.name} {time.perf_counter() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


KEY = ["detector", "family", "p1", "p2", "p3", "refractory"]
def _base(g):
    return ~g["detector"].str.contains(":")


def _prefix(g, p):
    return _base(g) | g["detector"].str.startswith(p + ":")


def _plain(g):
    return (g["family"] == "quantile") & (g["refractory"] == 0)


STAGES = {
    "A_quantile": lambda g: _base(g) & _plain(g),
    "B_quantile_refractory": lambda g: _base(g) & (g["family"] == "quantile"),
    "C_likelihood": lambda g: _base(g) & (g["family"] == "likelihood") & (g["refractory"] == 0),
    "D_likelihood_refractory": lambda g: _base(g) & (g["family"] == "likelihood"),
    "E_postprocessing": lambda g: _base(g),
    "F_log": lambda g: _prefix(g, "log") & _plain(g),
    "G_acf_context": lambda g: _prefix(g, "acf") & _plain(g),
    "H_all": lambda g: g["family"].notna(),
    "I_diverse_ensemble": lambda g: _prefix(g, "ens") & (g["family"] == "quantile"),
}


def complexity(cfg: tuple) -> int:
    _, family, p1, p2, p3, r = cfg
    extra = (p2 > 0) + (p3 > 0) if family == "quantile" else 1 + (p2 > 0)
    return int(extra + (r > 0))


def select(sub: pd.DataFrame, held: str, objective: str) -> tuple:
    train = sub[sub["category"] != held]
    piv = train.pivot_table(index=KEY, columns="case", values="f1")
    if objective == "median":
        return piv.median(axis=1).idxmax()
    stat = piv.mean(axis=1)
    if objective == "mean":
        return stat.idxmax()
    best = stat.idxmax()
    se = piv.loc[best].std() / np.sqrt(piv.shape[1])
    near = stat[stat >= stat[best] - se]
    return min(near.index, key=lambda c: (complexity(c), -near[c]))


def cross_validate(cases: list[Case], g: pd.DataFrame, objective: str, n_null: int, n_boot: int = 5000, seed: int = 0):
    by_name = {c.name: c for c in cases}
    rng = np.random.default_rng(seed)
    per_case, summary, base = [], [], None
    for stage, keep in STAGES.items():
        for scope in ("auto", *DETS, *DIVERSE):
            mask = keep(g)
            if not mask.any():
                continue
            sub = g[mask] if scope == "auto" else g[mask & (g["detector"].str.split(":").str[-1] == scope)]
            if sub.empty:
                continue
            rows, entries = [], []
            for held in sorted(g["category"].unique()):
                cfg = select(sub, held, objective)
                for case in [c for c in cases if c.category == held]:
                    a = alarms(case, cfg[0], cfg[1:])
                    t = case.t[case.test]
                    m = evaluate(t, a, case.windows, case.points, gap=case.w, grace=case.grace)
                    masks, in_window, after = regions(t, case.windows, case.grace)
                    entries.append((a, masks, ~in_window & ~after, len(masks), case.w))
                    rows.append({"stage": stage, "scope": scope, "case": case.name, "category": held,
                                 **dict(zip(KEY, cfg)), **{x: m[x] for x in ("events", "tp", "fp", "f1")}})
            r = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
            f1, chance, p = pooled(entries, n_null)
            rec = {"stage": stage, "scope": scope, "mean_f1": r["f1"].mean(), "median_f1": r["f1"].median(),
                   "pooled_f1": f1, "pooled_chance_f1": chance, "pooled_p": p,
                   "tp": r["tp"].sum(), "fp": r["fp"].sum(), "events": r["events"].sum()}
            if scope == "auto":
                base = r if base is None else base
                d = r["f1"].to_numpy() - base["f1"].to_numpy()
                boot = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
                rec.update(diff_vs_A=d.mean(), ci_low=np.quantile(boot, 0.025), ci_high=np.quantile(boot, 0.975),
                           chosen="; ".join(f"{k}: {v}" for k, v in (r["detector"] + " " + r["family"]).value_counts().items()))
            summary.append(rec)
            per_case.append(r)
            print(f"{stage:24s} {scope:18s} pooled={f1:.3f} chance={chance:.3f} tp={rec['tp']} fp={rec['fp']} "
                  f"mean={rec['mean_f1']:.3f}" + (f" diff={rec['diff_vs_A']:+.3f} [{rec['ci_low']:+.3f},{rec['ci_high']:+.3f}]"
                                                    if scope == "auto" else ""), flush=True)
    return pd.concat(per_case), pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["base"], choices=["base", "log", "acf"])
    parser.add_argument("--objective", default="mean", choices=["mean", "median", "1se"])
    parser.add_argument("--n-null", type=int, default=300)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    cases = []
    for name, spec in all_series().items():
        cases.append(build_case(name, spec, args.variants))
        print(f"  scored {name}", flush=True)
    path = RESULTS / "cv_grid.csv.gz"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["detector"])
    new = grid(cases, skip=set(old["detector"].unique()))
    g = pd.concat([old, new], ignore_index=True) if len(old) else new
    g.to_csv(path, index=False)
    per_case, summary = cross_validate(cases, g, args.objective, args.n_null)
    tag = f"_{args.tag}" if args.tag else f"_{args.objective}"
    per_case.to_csv(RESULTS / f"cv_per_case{tag}.csv", index=False)
    summary.to_csv(RESULTS / f"cv_summary{tag}.csv", index=False)
    print(summary[summary["scope"] == "auto"].drop(columns="chosen").round(3).to_string())


if __name__ == "__main__":
    main()
