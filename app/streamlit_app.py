import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import evaluate  # noqa: E402
from src.plots import threshold_view  # noqa: E402

RESULTS = ROOT / "results"
QS = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 1.0]
SKAB_FIT, SKAB_VAL = 300, 400

st.set_page_config(page_title="Sensor anomaly detection", layout="wide")


@st.cache_data
def nab():
    return pd.read_parquet(RESULTS / "scores.parquet"), json.loads((RESULTS / "meta.json").read_text())


@st.cache_data
def skab():
    return pd.read_parquet(RESULTS / "skab_scores.parquet"), json.loads((RESULTS / "skab_meta.json").read_text())


def stamps(pairs):
    return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in pairs]


st.title("Anomaly detection on sensor data")
st.caption(
    "Precomputed detector scores (seed 0) from the Numenta Anomaly Benchmark and SKAB. "
    "Move the threshold and see how alarms, false alarms and detection delay change."
)

dataset = st.sidebar.radio("Dataset", ["NAB (univariate)", "SKAB (multivariate)"])

if dataset.startswith("NAB"):
    scores, meta = nab()
    name = st.sidebar.selectbox("Series", list(meta))
    det = st.sidebar.selectbox("Detector", list(scores["detector"].unique()))
    df = scores[(scores["series"] == name) & (scores["detector"] == det)].reset_index(drop=True)
    m = meta[name]
    t = pd.DatetimeIndex(df["t"])
    windows = stamps(m["windows"])
    points = [pd.Timestamp(p) for p in m["points"]]
    normal = np.ones(len(t), dtype=bool)
    for a, b in windows:
        normal &= ~((t >= a) & (t <= b))
    val = (t >= pd.Timestamp(m["val_start"])) & (t < pd.Timestamp(m["test_start"])) & normal
    test = t >= pd.Timestamp(m["test_start"])
    gap, grace, unit = m["w"], pd.Timedelta(m["context"]), ("h", 1)
    contrib, value, value_name = None, df["x"].to_numpy(), "value"
    default_q = 0.995
else:
    scores, meta = skab()
    files = meta["files"]
    name = st.sidebar.selectbox("Experiment", list(files), format_func=lambda f: f"{f} · {files[f]['cause']}")
    det = st.sidebar.selectbox("Detector", list(scores["detector"].unique()))
    df = scores[(scores["file"] == name) & (scores["detector"] == det)].reset_index(drop=True)
    m = files[name]
    t = pd.DatetimeIndex(df["t"])
    windows = stamps(m["windows"])
    points = [a for a, _ in windows]
    idx = np.arange(len(t))
    val, test = (idx >= SKAB_FIT) & (idx < SKAB_VAL), idx >= SKAB_VAL
    gap, grace, unit = meta["w"], pd.Timedelta(seconds=meta["w"]), ("s", 3600)
    contrib = df[[c for c in df if c.startswith("c_")]].rename(columns=lambda c: c[2:])
    sensors = list(contrib.columns)
    value_name = st.sidebar.selectbox("Sensor to plot", sensors, index=sensors.index(m["plausible"][0]))
    value = df[value_name].to_numpy()
    default_q = meta["q"]

score = df["score"].to_numpy()
q = st.sidebar.select_slider(
    "Threshold (quantile of normal validation scores)", options=QS, value=default_q,
    help="1.0 = the largest score seen on normal validation data.",
)
thr = float(np.nanquantile(score[val], q))
alarms = np.nan_to_num(score, nan=-np.inf) > thr
st.sidebar.markdown(f"Threshold in score units: **{thr:.4g}**")

res = evaluate(t[test], alarms[test], windows, points, gap=gap, grace=grace, n_null=300)
delay = res["delay_h"] * unit[1]
cols = st.columns(6)
cols[0].metric("Event F1", f"{res['f1']:.2f}")
cols[1].metric("Chance F1", f"{res['chance_f1']:.2f}", help="Same alarms, shifted randomly in time (300 shifts).")
cols[2].metric("Detected events", f"{res['tp']} / {res['events']}")
cols[3].metric("False alarm runs", res["fp"])
cols[4].metric("Precision", f"{res['precision']:.2f}")
cols[5].metric("Median delay", "–" if np.isnan(delay) else f"{delay:.1f} {unit[0]}",
               help="From the labeled anomaly point; negative means early.")

st.plotly_chart(
    threshold_view(df["t"], score, thr, alarms, m["windows"], m["test_start"], value, value_name, contrib),
)
st.caption("Grey: labeled anomaly windows. Dotted line: start of the test part, only the test part is scored.")

if contrib is not None:
    st.subheader("Which sensor triggered the alarm?")
    inside = np.zeros(len(t), dtype=bool)
    for a, b in windows:
        inside |= (t >= a) & (t <= b)
    hit = alarms & inside & test
    if hit.any():
        share = contrib[hit].mean()
        share = (share / share.sum()).sort_values(ascending=False)
        left, right = st.columns([2, 1])
        left.bar_chart(share.rename("share of score"), horizontal=True)
        top = share.index[0]
        verdict = "plausible" if top in m["plausible"] else "not among the expected sensors"
        right.markdown(
            f"**Top sensor: {top}** ({share.iloc[0]:.0%})  \n"
            f"Fault: {m['cause']}  \nExpected sensors: {', '.join(m['plausible'])}  \nAssessment: {verdict}"
        )
        right.caption("Mean contribution over all alarms inside the labeled window.")
    else:
        st.info("No alarm inside the labeled anomaly window at this threshold.")
