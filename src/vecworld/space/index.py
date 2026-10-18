import numpy as np


def angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


class MeaningSpace:
    """Unit vectors for a fixed vocabulary, plus neighbour, step and 2D projection maths."""

    def __init__(self, words: list[str], vectors: np.ndarray):
        self.words = words
        self.vectors = np.asarray(vectors, np.float32)
        self._index = {w: i for i, w in enumerate(words)}

        self._mean = self.vectors.mean(axis=0)
        _, _, vt = np.linalg.svd(self.vectors - self._mean, full_matrices=False)
        self._components = vt[:2]
        coords = (self.vectors - self._mean) @ self._components.T
        self._scale = float(np.abs(coords).max()) or 1.0

    def vector(self, word: str) -> np.ndarray:
        return self.vectors[self._index[word]]

    def neighbours(self, p: np.ndarray, k: int, exclude: set[str] = frozenset()) -> list[tuple[str, float]]:
        sims = self.vectors @ p
        out = []
        for i in np.argsort(-sims):
            if self.words[i] in exclude:
                continue
            out.append((self.words[i], float(sims[i])))
            if len(out) == k:
                break
        return out

    def step_toward(self, p: np.ndarray, target: np.ndarray, max_angle: float) -> np.ndarray:
        theta = angle(p, target)
        if theta < 1e-6:
            return target.copy()
        t = min(1.0, max_angle / theta)
        s = np.sin(theta)
        out = (np.sin((1 - t) * theta) * p + np.sin(t * theta) * target) / s
        return (out / np.linalg.norm(out)).astype(np.float32)

    def heading(self, p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray | None:
        """Unit tangent at p_to along the great circle from p_from, pointing onward (None if no direction)."""
        t = float(np.dot(p_from, p_to)) * p_to - p_from
        n = float(np.linalg.norm(t))
        return None if n < 1e-6 else (t / n).astype(np.float32)

    def walk(self, p: np.ndarray, h: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
        """Move `angle` along the great circle through p with tangent h; returns the new point and tangent."""
        c, s = np.cos(angle), np.sin(angle)
        p2, h2 = c * p + s * h, -s * p + c * h
        return (p2 / np.linalg.norm(p2)).astype(np.float32), (h2 / np.linalg.norm(h2)).astype(np.float32)

    def project(self, p: np.ndarray) -> tuple[float, float]:
        x, y = (p - self._mean) @ self._components.T / self._scale
        return float(x), float(y)
