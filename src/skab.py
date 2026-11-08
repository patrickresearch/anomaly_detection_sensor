import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import RAW

URL = "https://raw.githubusercontent.com/waico/SKAB/master/data"
COLUMNS = {
    "Accelerometer1RMS": "accel1",
    "Accelerometer2RMS": "accel2",
    "Current": "current",
    "Pressure": "pressure",
    "Temperature": "motor_temp",
    "Thermocouple": "fluid_temp",
    "Voltage": "voltage",
    "Volume Flow RateRMS": "flow",
}
SENSORS = list(COLUMNS.values())
FILES = [f"valve1/{i}" for i in range(16)] + [f"valve2/{i}" for i in range(4)] + [f"other/{i}" for i in range(1, 15)]

HYDRAULIC = {"flow", "pressure"}
VIBRATION = {"accel1", "accel2"}
# Fault description from the SKAB data README and the sensors a fault of that kind should move first.
# This is a domain assumption used as a plausibility check, not a ground-truth label.
CAUSES = {
    "valve1": ("inlet valve closed", HYDRAULIC),
    "valve2": ("outlet valve closed", HYDRAULIC),
    **{f"other/{i}": ("fluid leak or addition", HYDRAULIC) for i in range(1, 5)},
    **{f"other/{i}": ("rotor imbalance", VIBRATION) for i in range(5, 10)},
    "other/10": ("slow water increase", HYDRAULIC),
    "other/11": ("sudden water increase", HYDRAULIC),
    "other/12": ("draining until cavitation", HYDRAULIC | VIBRATION | {"current"}),
    "other/13": ("two-phase inlet flow (cavitation)", HYDRAULIC | VIBRATION | {"current"}),
    "other/14": ("hot water supply", {"fluid_temp"}),
}


@dataclass
class Experiment:
    name: str
    t: pd.DatetimeIndex
    X: np.ndarray
    y: np.ndarray

    @property
    def cause(self) -> tuple[str, set[str]]:
        return CAUSES[self.name] if self.name in CAUSES else CAUSES[self.name.split("/")[0]]

    def windows(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        y = np.r_[False, self.y, False]
        starts, ends = np.flatnonzero(~y[:-1] & y[1:]), np.flatnonzero(y[:-1] & ~y[1:]) - 1
        return [(self.t[a], self.t[b]) for a, b in zip(starts, ends)]


def fetch(name: str) -> Path:
    local = RAW / "skab" / f"{name}.csv"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{URL}/{name}.csv", local)
    return local


def load(name: str) -> Experiment:
    df = pd.read_csv(fetch(name), sep=";", parse_dates=["datetime"]).sort_values("datetime")
    y = df["anomaly"].to_numpy(dtype=bool) if "anomaly" in df else np.zeros(len(df), dtype=bool)
    return Experiment(name, pd.DatetimeIndex(df["datetime"]), df[list(COLUMNS)].to_numpy(dtype=float), y)
