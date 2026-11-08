import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {"zscore": "#2a78d6", "iforest": "#eb6834", "matrix_profile": "#1baf7a", "autoencoder": "#6250d6"}
SYMBOLS = {"zscore": "circle", "iforest": "diamond", "matrix_profile": "square", "autoencoder": "triangle-up"}
LINE, BAND, INK = "#52514e", "rgba(82,81,78,0.15)", "#0b0b0b"


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
        margin=dict(l=60, r=20, t=80, b=40),
    )


def hero(scores: pd.DataFrame, meta: dict, series: str) -> go.Figure:
    df = scores[scores["series"] == series]
    base = df[df["detector"] == df["detector"].iloc[0]]
    fig = go.Figure(go.Scatter(x=base["t"], y=base["x"], name="value", line=dict(color=LINE, width=1.5)))
    for det, g in df.groupby("detector", sort=False):
        o = onsets(g)
        fig.add_scatter(
            x=o["t"], y=o["x"], mode="markers", name=det,
            marker=dict(color=COLORS[det], symbol=SYMBOLS[det], size=10, line=dict(color="white", width=2)),
        )
    decorate(fig, meta[series])
    return layout(fig, f"{series}: labeled anomaly windows (grey) and alarm onsets per detector", 480)


def small_multiples(scores: pd.DataFrame, meta: dict, series: str) -> go.Figure:
    df = scores[scores["series"] == series]
    dets = list(df["detector"].unique())
    fig = make_subplots(rows=len(dets), cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=dets)
    for i, det in enumerate(dets, 1):
        g = df[df["detector"] == det]
        a = g[g["alarm"]]
        fig.add_scatter(x=g["t"], y=g["x"], line=dict(color=LINE, width=1), showlegend=False, row=i, col=1)
        fig.add_scatter(
            x=a["t"], y=a["x"], mode="markers", name=det, showlegend=False,
            marker=dict(color=COLORS[det], symbol=SYMBOLS[det], size=8), row=i, col=1,
        )
        decorate(fig, meta[series], row=i, col=1)
    return layout(fig, f"{series}: alarms per detector", 220 * len(dets))
