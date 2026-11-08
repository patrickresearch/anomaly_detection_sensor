import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "zscore": "#2a78d6",
    "iforest": "#eb6834",
    "matrix_profile": "#1baf7a",
    "matrix_profile_raw": "#eda100",
    "autoencoder": "#e87ba4",
    "ensemble": "#008300",
}
SYMBOLS = {
    "zscore": "circle",
    "iforest": "diamond",
    "matrix_profile": "square",
    "matrix_profile_raw": "pentagon",
    "autoencoder": "triangle-up",
    "ensemble": "star",
}
LINE, BAND, INK = "#52514e", "rgba(82,81,78,0.15)", "#0b0b0b"


def thin(df: pd.DataFrame, n: int = 1500) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    if len(df) <= 2 * n:
        return df
    g = df["x"].groupby(np.arange(len(df)) * n // len(df))
    return df.loc[np.union1d(g.idxmin().to_numpy(), g.idxmax().to_numpy())]


def onsets(df: pd.DataFrame) -> pd.DataFrame:
    a = df["alarm"]
    return df[a & ~a.shift(fill_value=False)]


def decorate(fig: go.Figure, meta: dict, **rc) -> None:
    for a, b in meta["windows"]:
        fig.add_vrect(x0=a, x1=b, fillcolor=BAND, line_width=0, layer="below", **rc)
    fig.add_vline(x=meta["test_start"], line=dict(color=LINE, dash="dot", width=1), **rc)


def layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    return fig.update_layout(
        title=title, height=height, template="plotly_white", hovermode="x unified",
        font=dict(color=INK), legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
        margin=dict(l=140, r=20, t=80, b=40),
    )


def hero(scores: pd.DataFrame, meta: dict, series: str) -> go.Figure:
    df = scores[scores["series"] == series]
    dets = list(df["detector"].unique())
    fig = make_subplots(
        rows=1 + len(dets), cols=1, shared_xaxes=True, vertical_spacing=0.015,
        row_heights=[0.55] + [0.45 / len(dets)] * len(dets),
    )
    base = thin(df[df["detector"] == dets[0]])
    fig.add_scatter(x=base["t"], y=base["x"], name="value", line=dict(color=LINE, width=1.5), row=1, col=1)
    decorate(fig, meta[series], row=1, col=1)
    for i, det in enumerate(dets, 2):
        o = onsets(df[df["detector"] == det])
        fig.add_scatter(
            x=o["t"], y=np.zeros(len(o)), mode="markers", name=det, showlegend=False,
            marker=dict(color=COLORS[det], symbol=SYMBOLS[det], size=11, line=dict(color="white", width=2)),
            row=i, col=1,
        )
        fig.update_yaxes(
            tickvals=[0], ticktext=[det], range=[-1, 1], showgrid=False, zeroline=False, fixedrange=True, row=i, col=1,
        )
        decorate(fig, meta[series], row=i, col=1)
    fig.update_yaxes(title_text="value", row=1, col=1)
    return layout(fig, f"{series}: labeled anomaly windows (grey) and alarm onsets per detector", 300 + 48 * len(dets))


def small_multiples(scores: pd.DataFrame, meta: dict, series: str) -> go.Figure:
    df = scores[scores["series"] == series]
    dets = list(df["detector"].unique())
    fig = make_subplots(rows=len(dets), cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=dets)
    for i, det in enumerate(dets, 1):
        g = df[df["detector"] == det]
        a = g[g["alarm"]]
        line = thin(g)
        fig.add_scatter(x=line["t"], y=line["x"], line=dict(color=LINE, width=1), showlegend=False, row=i, col=1)
        fig.add_scatter(
            x=a["t"], y=a["x"], mode="markers", name=det, showlegend=False,
            marker=dict(color=COLORS[det], symbol=SYMBOLS[det], size=8), row=i, col=1,
        )
        decorate(fig, meta[series], row=i, col=1)
    return layout(fig, f"{series}: alarms per detector", 200 * len(dets))


def sweep_chart(sweep: pd.DataFrame) -> go.Figure:
    m = sweep.groupby(["detector", "q"], sort=False)["f1"].mean().reset_index()
    fig = go.Figure()
    for det, g in m.groupby("detector", sort=False):
        fig.add_scatter(
            x=1 - g["q"], y=g["f1"], mode="lines+markers", name=det,
            line=dict(color=COLORS[det], width=2), marker=dict(symbol=SYMBOLS[det], size=9, color=COLORS[det]),
            hovertemplate="q=%{customdata}<br>mean F1=%{y:.2f}", customdata=g["q"],
        )
    fig.update_xaxes(type="log", autorange="reversed", title="1 − q (share of normal validation points above threshold)")
    fig.update_yaxes(title="mean event F1", rangemode="tozero")
    fig = layout(fig, "Threshold sensitivity: mean event F1 over series and seeds", 420)
    return fig.update_layout(hovermode="closest", margin=dict(l=60))


CRITICAL = "#d03b3b"
RAMP = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]


def threshold_view(
    t: pd.Series,
    score: np.ndarray,
    thr: float,
    alarms: np.ndarray,
    windows: list,
    test_start: str,
    value: np.ndarray | None = None,
    value_name: str = "value",
    contrib: pd.DataFrame | None = None,
) -> go.Figure:
    rows = 2 + (contrib is not None)
    heights = [0.45, 0.35, 0.2] if contrib is not None else [0.55, 0.45]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=heights)
    meta = {"windows": windows, "test_start": test_start}
    line = thin(pd.DataFrame({"t": t, "x": value}))
    fig.add_scattergl(x=line["t"], y=line["x"], name=value_name, line=dict(color=LINE, width=1), row=1, col=1)
    fig.add_scattergl(
        x=t[alarms], y=value[alarms], mode="markers", name="alarm",
        marker=dict(color=CRITICAL, size=5, symbol="triangle-up"), row=1, col=1,
    )
    s = thin(pd.DataFrame({"t": t, "x": score}).dropna())
    fig.add_scattergl(x=s["t"], y=s["x"], name="score", line=dict(color=INK, width=1), row=2, col=1)
    fig.add_hline(y=thr, line=dict(color=LINE, dash="dash", width=1), row=2, col=1,
                  annotation_text="threshold", annotation_position="top left")
    for r in range(1, rows + 1):
        decorate(fig, meta, row=r, col=1)
    fig.update_yaxes(title_text=value_name, row=1, col=1)
    fig.update_yaxes(title_text="score (log)", type="log", row=2, col=1)
    if contrib is not None:
        share = contrib.div(contrib.sum(1), axis=0)
        fig.add_heatmap(
            x=t, y=list(contrib.columns), z=share.T.to_numpy(), zmin=0, zmax=0.5, colorscale=RAMP,
            colorbar=dict(title="share", len=0.2, y=0.1, thickness=10), row=3, col=1,
            hovertemplate="%{y}: %{z:.0%}<extra></extra>",
        )
    fig = layout(fig, "", 300 + 220 * (rows - 1))
    return fig.update_layout(margin=dict(l=60, r=20, t=30, b=30), hovermode="closest")


def attribution_heatmap(attribution: pd.DataFrame, detector: str, sensors: list[str]) -> go.Figure:
    a = attribution[attribution["detector"] == detector]
    share = pd.crosstab(a["cause"], a["top"], normalize="index").reindex(columns=sensors, fill_value=0)
    n = a.groupby("cause").size()
    fig = go.Figure(go.Heatmap(
        z=share.to_numpy(), x=sensors, y=[f"{c} (n={n[c]})" for c in share.index], zmin=0, zmax=1,
        colorscale=RAMP, colorbar=dict(title="share", thickness=10),
        text=share.map(lambda v: f"{v:.0%}" if v else "").to_numpy(), texttemplate="%{text}",
        hovertemplate="%{y}<br>top sensor %{x}: %{z:.0%}<extra></extra>",
    ))
    fig = layout(fig, f"{detector}: top sensor per detected event, by fault type", 420)
    return fig.update_layout(margin=dict(l=240), hovermode="closest")
