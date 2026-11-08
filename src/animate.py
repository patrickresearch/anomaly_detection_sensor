import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

from src.data import ROOT
from src.multivariate import MV_DETECTORS
from src.run_skab import FIT, VAL
from src.skab import SENSORS, load
from src.streaming import replay

RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, MUTED, LINE, BAND, CRITICAL, SURFACE = "#0b0b0b", "#6b6a66", "#52514e", "#e6e5e1", "#d03b3b", "#fcfcfb"


def stream(name: str, detector: str, w: int, q: float):
    e = load(name)
    model = MV_DETECTORS[detector](w, 0).fit(e.X[:FIT])
    thr = float(np.nanquantile(model.score(e.X[:VAL])[0][FIT:VAL], q))
    t0 = time.perf_counter()
    score, C, alarm = replay(model, thr, e.X)
    latency = (time.perf_counter() - t0) / len(e.X) * 1e3
    batch = model.score(e.X)[0]
    info = {
        "file": name, "cause": e.cause[0], "detector": detector, "w": w, "q": q, "threshold": thr,
        "points": len(e.X), "ms_per_point": latency,
        "max_rel_diff_to_batch": float(np.nanmax(np.abs(score - batch) / np.abs(batch))),
    }
    return e, score, C, alarm, thr, info


def animate(e, score, C, alarm, thr, info, out, step: int, fps: int) -> None:
    n = len(score)
    x = np.arange(n)
    share = (C / C.sum(1, keepdims=True)).T
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": BAND})
    fig = plt.figure(figsize=(10, 5.4), dpi=80, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.35], height_ratios=[1, 1.1], hspace=0.1, wspace=0.32,
                          left=0.09, right=0.97, top=0.86, bottom=0.1)
    ax_s, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[:, 1])
    ax_h = fig.add_subplot(gs[1, 0], sharex=ax_s)
    for ax in (ax_s, ax_h, ax_b):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(f"Streaming detection on SKAB {info['file']} ({info['cause']}), {info['detector']}, one point per second",
                 x=0.09, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.09, 0.905, "grey: labeled anomaly    red ticks: alarms    dotted: end of training and threshold calibration", color=MUTED)

    y = e.y.astype(int)
    starts, ends = np.flatnonzero(np.diff(np.r_[0, y]) == 1), np.flatnonzero(np.diff(np.r_[y, 0]) == -1)
    for a, b in zip(starts, ends):
        ax_s.axvspan(a, b, color=BAND, lw=0, zorder=0)
    for v in (FIT, VAL):
        ax_s.axvline(v, color=LINE, ls=":", lw=1)
        ax_h.axvline(v, color=LINE, ls=":", lw=1)
    finite = score[np.isfinite(score) & (score > 0)]
    lo, hi = finite.min() * 0.8, finite.max() * 3
    ax_s.set_yscale("log")
    ax_s.set_ylim(lo, hi)
    ax_s.set_xlim(0, n)
    ax_s.axhline(thr, color=LINE, ls="--", lw=1)
    ax_s.text(n * 0.005, thr * 1.15, "threshold", color=MUTED, fontsize=8)
    ax_s.set_ylabel("anomaly score")
    ax_s.tick_params(labelbottom=False)
    line, = ax_s.plot([], [], color=INK, lw=1.2)
    rug = ax_s.scatter([], [], marker="|", s=60, color=CRITICAL, lw=1.2)
    cur_s = ax_s.axvline(0, color=INK, lw=0.8, alpha=0.4)

    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
    cmap.set_bad(SURFACE)
    im = ax_h.imshow(np.full_like(share, np.nan), aspect="auto", cmap=cmap, vmin=0, vmax=0.5, interpolation="nearest",
                     extent=(0, n, len(SENSORS) - 0.5, -0.5))
    for a, b in zip(starts, ends):
        ax_h.axvline(a, color=LINE, lw=0.8)
        ax_h.axvline(b, color=LINE, lw=0.8)
    ax_h.set_yticks(range(len(SENSORS)), SENSORS)
    ax_h.set_xlabel("seconds")
    ax_h.set_title("share of the score per sensor (darker = larger, capped at 50%)", loc="left", fontsize=9, color=MUTED)
    cur_h = ax_h.axvline(0, color=INK, lw=0.8, alpha=0.4)

    bars = ax_b.barh(range(len(SENSORS)), np.zeros(len(SENSORS)), color=RAMP[1], height=0.7)
    ax_b.set_yticks(range(len(SENSORS)), SENSORS)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("share of current score (%)")
    labels = [ax_b.text(0, i, "", va="center", fontsize=8, color=INK) for i in range(len(SENSORS))]
    icon = ax_b.text(0, 1.04, "", transform=ax_b.transAxes, fontsize=11, va="bottom")
    status = ax_b.text(0.09, 1.04, "", transform=ax_b.transAxes, fontsize=9, va="bottom", color=INK)

    def update(i):
        line.set_data(x[: i + 1], score[: i + 1])
        hit = np.flatnonzero(alarm[: i + 1])
        rug.set_offsets(np.c_[hit, np.full(len(hit), hi * 0.6)] if len(hit) else np.empty((0, 2)))
        data = share.copy()
        data[:, i + 1:] = np.nan
        im.set_data(np.ma.masked_invalid(data))
        cur_s.set_xdata([i, i])
        cur_h.set_xdata([i, i])
        now = share[:, i] * 100 if np.isfinite(share[:, i]).all() else np.zeros(len(SENSORS))
        top = int(np.argmax(now))
        for k, (bar, lab) in enumerate(zip(bars, labels)):
            bar.set_width(now[k])
            bar.set_color(RAMP[3] if k == top and alarm[i] else RAMP[1])
            lab.set_text(f"{now[k]:.0f}%")
            lab.set_x(now[k] + 1.5)
        if alarm[i]:
            icon.set_text("▲")
            icon.set_color(CRITICAL)
            status.set_text(f"alarm, top sensor {SENSORS[top]}")
        else:
            icon.set_text("●")
            icon.set_color(MUTED)
            status.set_text("normal" if i >= VAL else "calibrating")
        return line, rug, im, cur_s, cur_h, *bars, *labels, icon, status

    frames = list(range(info["w"], n, step)) + [n - 1] * fps * 2
    FuncAnimation(fig, update, frames=frames).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="other/7")
    parser.add_argument("--detector", default="autoencoder", choices=list(MV_DETECTORS))
    parser.add_argument("--w", type=int, default=20)
    parser.add_argument("--q", type=float, default=0.99)
    parser.add_argument("--step", type=int, default=5, help="points per frame")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out", default=str(ROOT / "docs" / "streaming.gif"))
    args = parser.parse_args()

    e, score, C, alarm, thr, info = stream(args.file, args.detector, args.w, args.q)
    print(json.dumps(info, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "streaming.json").write_text(json.dumps(info, indent=2))
    animate(e, score, C, alarm, thr, info, args.out, args.step, args.fps)


if __name__ == "__main__":
    main()
