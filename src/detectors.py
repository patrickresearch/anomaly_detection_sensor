from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import stumpy
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
    def fit(self, x):
        self.reference = x
        return self

    def score(self, x):
        mp = stumpy.stump(x, self.w, T_B=self.reference, ignore_trivial=False)
        return align(mp[:, 0].astype(float), len(x))


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, 8, 7, stride=2, padding=3), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))[..., : x.shape[-1]]


class Autoencoder(Detector):
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


DETECTORS = {
    "zscore": RollingZScore,
    "iforest": IForest,
    "matrix_profile": MatrixProfile,
    "autoencoder": Autoencoder,
}
