"""Real backend tests. Skips mean NOT VALIDATED, not passing model validation."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'PyTorch unavailable; real model not validated')
class ModernBackendTests(unittest.TestCase):
    def setUp(self):
        from pad_omni.backend import Backend
        self.Backend = Backend
        self.c = dict(hidden=6, dense=6, z_dim=2, nf_layers=2, window_length=4,
                      batch_size=3, max_epoch=1, n_z=2, seed=42, torch_threads=1)
        self.values = np.random.default_rng(42).normal(size=(60, 2)).astype('float32')

    def test_fit_restore_and_batch_invariance(self):
        a = self.Backend(self.c, 2)
        a.fit(self.values)
        scores = a.scores(self.values)
        self.assertEqual(len(scores), 57)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores[-1:], a.scores(self.values[-4:]), rtol=1e-5, atol=1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'model'
            a.save(prefix)
            b = self.Backend(dict(self.c, batch_size=5), 2)
            b.restore(prefix)
            np.testing.assert_allclose(scores, b.scores(self.values), rtol=1e-5, atol=1e-5)
            b.close()
        a.close()

    def test_bad_input_and_checkpoint(self):
        a = self.Backend(self.c, 2)
        with self.assertRaises(ValueError):
            a.scores(np.full((4, 2), np.nan))
        with self.assertRaises(ValueError):
            a.scores(self.values[:2])
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'model'
            a.save(prefix)
            b = self.Backend(dict(self.c, hidden=7), 2)
            with self.assertRaises(ValueError):
                b.restore(prefix)
            b.close()
        a.close()


if __name__ == '__main__':
    unittest.main()
