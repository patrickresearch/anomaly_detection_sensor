import copy
from collections import deque

import numpy as np


class Stream:

    def __init__(self, detector, threshold: float):
        self.alpha = getattr(detector, "alpha", None)
        self.detector = detector.inner if self.alpha else detector
        self.stateful = hasattr(self.detector, "step")
        if self.stateful:
            self.detector = copy.deepcopy(self.detector)
        self.threshold = threshold
        self.buffer = deque(maxlen=detector.w + 1)
        self.base = None

    def update(self, x) -> tuple[float, np.ndarray | None, bool]:
        if self.alpha:
            x = np.asarray(x, dtype=float)
            r = np.zeros_like(x) if self.base is None else x - self.base
            self.base = x if self.base is None else (1 - self.alpha) * self.base + self.alpha * x
            x = r
        self.buffer.append(x)
        if self.stateful:
            score = self.detector.step(np.asarray(self.buffer))
            return score, None, bool(score > self.threshold)
        if len(self.buffer) < self.buffer.maxlen:
            return np.nan, None, False
        out = self.detector.score(np.asarray(self.buffer))
        score, contrib = (out[0][-1], out[1][-1]) if isinstance(out, tuple) else (out[-1], None)
        return float(score), contrib, bool(score > self.threshold)


def replay(detector, threshold: float, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    stream = Stream(detector, threshold)
    out = [stream.update(x) for x in X]
    scores = np.array([o[0] for o in out])
    contribs = None if X.ndim == 1 else np.array([o[1] if o[1] is not None else np.full(X.shape[1], np.nan) for o in out])
    return scores, contribs, np.array([o[2] for o in out])
