import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
NAB = "https://raw.githubusercontent.com/numenta/NAB/master"

SERIES = {
    "machine_temperature": ("realKnownCause/machine_temperature_system_failure.csv", "12h", False),
    "nyc_taxi": ("realKnownCause/nyc_taxi.csv", "1D", True),
    "ambient_temperature": ("realKnownCause/ambient_temperature_system_failure.csv", "1D", True),
    "ec2_request_latency": ("realKnownCause/ec2_request_latency_system_failure.csv", "2h", False),
    "cpu_asg_misconfig": ("realKnownCause/cpu_utilization_asg_misconfiguration.csv", "2h", False),
    "rds_cpu": ("realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv", "2h", False),
    "ec2_cpu": ("realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv", "2h", False),
    "traffic_occupancy": ("realTraffic/occupancy_t4013.csv", "2h", False),
    "traffic_speed": ("realTraffic/speed_7578.csv", "2h", False),
}


@dataclass
class Series:
    name: str
    t: pd.DatetimeIndex
    x: np.ndarray
    windows: list[tuple[pd.Timestamp, pd.Timestamp]]
    points: list[pd.Timestamp]
    w: int
    context: pd.Timedelta
    weekly: bool

    def in_window(self) -> np.ndarray:
        mask = np.zeros(len(self.t), dtype=bool)
        for start, end in self.windows:
            mask |= (self.t >= start) & (self.t <= end)
        return mask


def fetch(path: str) -> Path:
    local = RAW / path
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{NAB}/{path}", local)
    return local


def load(name: str) -> Series:
    path, context, weekly = SERIES[name]
    df = pd.read_csv(fetch(f"data/{path}"), parse_dates=["timestamp"])
    windows = json.loads(fetch("labels/combined_windows.json").read_text())[path]
    points = json.loads(fetch("labels/combined_labels.json").read_text())[path]
    t = pd.DatetimeIndex(df["timestamp"])
    w = int(pd.Timedelta(context) / t.to_series().diff().median())
    return Series(
        name=name,
        t=t,
        x=df["value"].to_numpy(dtype=float),
        windows=[(pd.Timestamp(a), pd.Timestamp(b)) for a, b in windows],
        points=[pd.Timestamp(p) for p in points],
        w=w,
        context=pd.Timedelta(context),
        weekly=weekly,
    )


def remove_weekly(t: pd.DatetimeIndex, x: np.ndarray, fit: np.ndarray) -> np.ndarray:
    key = pd.Index(t.dayofweek * 86400 + (t - t.normalize()).total_seconds().astype(int))
    profile = pd.Series(x[fit]).groupby(key[fit]).median()
    return x - profile.reindex(key).fillna(np.median(x[fit])).to_numpy()


def split(n: int, train: float, val: float) -> tuple[slice, slice, slice]:
    a, b = int(n * train), int(n * (train + val))
    return slice(0, a), slice(a, b), slice(b, n)
