import numpy as np


class Standardizer:
    def fit(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError('finite 2-D training data required')
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean):
            raise ValueError('scaler feature mismatch')
        z = ((x - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError('nonfinite transformed input')
        return z

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as d:
            obj.mean, obj.scale = d['mean'], d['scale']
        if (obj.mean.ndim != 1 or obj.scale.shape != obj.mean.shape or
                not np.isfinite(obj.mean).all() or not np.isfinite(obj.scale).all()
                or (obj.scale <= 0).any()):
            raise ValueError('invalid scaler artifact')
        return obj


def window_batches(values, window, batch_size):
    """Reference 3-D windowing for tests/adapters, not for Trainer.fit()."""
    if window < 1 or batch_size < 1 or len(values) < window:
        raise ValueError('invalid window/batch/data length')
    for start in range(0, len(values)-window+1, batch_size):
        ends = range(start+window, min(start+batch_size+window, len(values)+1))
        yield np.stack([values[end-window:end] for end in ends])
