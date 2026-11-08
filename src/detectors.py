from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import stumpy
from scipy import stats
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest
from torch import nn


def windows(x: np.ndarray, w: int) -> np.ndarray:
    return sliding_window_view(x, w)


def align(s: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    out[n - len(s):] = s
    return out


class Detector(ABC):
    stochastic = False

    def __init__(self, w: int, seed: int = 0):
        self.w = w
        self.seed = seed

    def fit(self, x: np.ndarray) -> "Detector":
        return self

    @abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray: ...


class RollingZScore(Detector):
    def score(self, x):
        s = pd.Series(x)
        past = s.shift(1).rolling(self.w)
        return ((s - past.mean()) / past.std()).abs().to_numpy()


class IForest(Detector):
    stochastic = True

    @staticmethod
    def features(W: np.ndarray) -> np.ndarray:
        t = np.arange(W.shape[1]) - (W.shape[1] - 1) / 2
        mean = W.mean(1)
        return np.column_stack([
            mean,
            W.std(1),
            W.min(1),
            W.max(1),
            W[:, -1],
            W[:, -1] - mean,
            W @ t / (t @ t),
            np.diff(W, axis=1).std(1),
        ])

    def fit(self, x):
        W = windows(x, self.w)
        W = W[np.isfinite(W).all(1)]
        self.model = IsolationForest(n_estimators=300, random_state=self.seed).fit(self.features(W))
        return self

    def score(self, x):
        return align(-self.model.score_samples(self.features(windows(x, self.w))), len(x))


class MatrixProfile(Detector):
    normalize = True

    def fit(self, x):
        self.reference = x
        return self

    def score(self, x):
        mp = stumpy.stump(x, self.w, T_B=self.reference, ignore_trivial=False, normalize=self.normalize)
        return align(mp[:, 0].astype(float), len(x))


class MatrixProfileRaw(MatrixProfile):
    normalize = False


class ConvAE(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 16, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, 8, 7, stride=2, padding=3), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, channels, 7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))[..., : x.shape[-1]]


class Autoencoder(Detector):
    stochastic = True

    def __init__(self, w: int, seed: int = 0, epochs: int = 30, batch: int = 128, lr: float = 1e-3):
        super().__init__(w, seed)
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def tensor(self, W: np.ndarray) -> torch.Tensor:
        return torch.tensor((W - self.mu) / self.sd, dtype=torch.float32).unsqueeze(1)

    def fit(self, x):
        torch.manual_seed(self.seed)
        self.mu, self.sd = np.nanmean(x), np.nanstd(x)
        W = windows(x, self.w)
        X = self.tensor(W[np.isfinite(W).all(1)])
        self.model = ConvAE().to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loader = torch.utils.data.DataLoader(X, batch_size=self.batch, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for xb in loader:
                xb = xb.to(self.device)
                loss = nn.functional.mse_loss(self.model(xb), xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, x):
        self.model.eval()
        X = self.tensor(windows(x, self.w))
        err = []
        for xb in torch.split(X, 1024):
            xb = xb.to(self.device)
            err.append(((self.model(xb) - xb) ** 2).mean((1, 2)).cpu())
        return align(torch.cat(err).numpy(), len(x))


class RelativeEntropy(Detector):
    bins = 5
    c_th = 1

    def fit(self, x):
        x = x[np.isfinite(x)]
        lo, hi = x.min(), x.max()
        self.edges = np.unique(np.r_[lo, np.linspace(lo, hi, self.bins + 1)[1:-1], hi])
        self.P, self.c = np.empty((0, len(self.edges) + 1)), np.empty(0)
        self.T = stats.chi2.isf(0.01, len(self.edges))
        self._run(x)
        return self

    def _hist(self, b: np.ndarray) -> np.ndarray:
        h = np.bincount(b, minlength=len(self.edges) + 1) + 0.5
        return h / h.sum()

    def step(self, window: np.ndarray) -> float:
        if len(window) < self.w or not np.isfinite(window).all():
            return np.nan
        p = self._hist(np.digitize(window[-self.w:], self.edges))
        stat = 2 * self.w * (p * np.log(p / self.P)).sum(1) if len(self.P) else np.empty(0)
        common = self.c > self.c_th
        out = stat[common].min() if common.any() else self.T * 2
        if len(stat) and stat.min() < self.T:
            self.c[stat.argmin()] += 1
        else:
            self.P, self.c = np.vstack([self.P, p]), np.append(self.c, 1)
        return float(out)

    def _run(self, x: np.ndarray) -> np.ndarray:
        out = np.full(len(x), np.nan)
        for t in range(self.w - 1, len(x)):
            out[t] = self.step(x[t - self.w + 1:t + 1])
        return out

    def score(self, x):
        P, c = self.P.copy(), self.c.copy()
        out = self._run(np.asarray(x, dtype=float))
        self.P, self.c = P, c
        return out


DETECTORS = {
    "zscore": RollingZScore,
    "iforest": IForest,
    "matrix_profile": MatrixProfile,
    "matrix_profile_raw": MatrixProfileRaw,
    "autoencoder": Autoencoder,
    "relative_entropy": RelativeEntropy,
}
