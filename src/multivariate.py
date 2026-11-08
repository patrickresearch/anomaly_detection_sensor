import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from src.detectors import ConvAE


def trailing_mean(C: np.ndarray, w: int) -> np.ndarray:
    cs = np.vstack([np.zeros((1, C.shape[1])), np.cumsum(C, 0)])
    out = np.full(C.shape, np.nan)
    out[w - 1:] = (cs[w:] - cs[:-w]) / w
    return out


class MvDetector:
    stochastic = False

    def __init__(self, w: int, seed: int = 0):
        self.w = w
        self.seed = seed

    def fit(self, X: np.ndarray) -> "MvDetector":
        self.mu, self.sd = X.mean(0), np.maximum(X.std(0), 1e-9)
        self._fit(self.standardize(X))
        return self

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def _fit(self, Z: np.ndarray) -> None:
        pass

    def contributions(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        C = trailing_mean(self.contributions(self.standardize(X)), self.w)
        return C.sum(1), C


class MvZScore(MvDetector):
    def contributions(self, Z):
        return Z**2


class PCA(MvDetector):
    def __init__(self, w: int, seed: int = 0, explained: float = 0.9):
        super().__init__(w, seed)
        self.explained = explained

    def _fit(self, Z):
        lam, P = np.linalg.eigh(np.cov(Z, rowvar=False))
        lam, P = lam[::-1], P[:, ::-1]
        k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), self.explained)) + 1
        Pk = P[:, :k]
        T2 = Pk @ np.diag(1 / lam[:k]) @ Pk.T
        Q = np.eye(len(lam)) - Pk @ Pk.T
        # scale both parts by their training mean so neither dominates the combined index
        M = T2 / np.einsum("ij,jk,ik->i", Z, T2, Z).mean() + Q / np.einsum("ij,jk,ik->i", Z, Q, Z).mean()
        e, V = np.linalg.eigh(M)
        self.root = V @ np.diag(np.sqrt(np.clip(e, 0, None))) @ V.T
        self.k = k

    def contributions(self, Z):
        return (Z @ self.root) ** 2


class MvAutoencoder(MvDetector):
    stochastic = True

    def __init__(self, w: int, seed: int = 0, epochs: int = 60, batch: int = 64, lr: float = 1e-3):
        super().__init__(w, seed)
        self.epochs, self.batch, self.lr = epochs, batch, lr

    def windows(self, Z: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.ascontiguousarray(sliding_window_view(Z, self.w, axis=0)), dtype=torch.float32)

    def _fit(self, Z):
        torch.manual_seed(self.seed)
        X = self.windows(Z)
        self.model = ConvAE(channels=Z.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loader = torch.utils.data.DataLoader(X, batch_size=self.batch, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for xb in loader:
                loss = nn.functional.mse_loss(self.model(xb), xb)
                opt.zero_grad()
                loss.backward()
                opt.step()

    @torch.no_grad()
    def score(self, X):
        self.model.eval()
        W = self.windows(self.standardize(X))
        err = ((self.model(W) - W) ** 2).mean(2).numpy()
        C = np.full((len(X), X.shape[1]), np.nan)
        C[self.w - 1:] = err
        return C.sum(1), C


MV_DETECTORS = {
    "zscore": MvZScore,
    "pca": PCA,
    "autoencoder": MvAutoencoder,
}
