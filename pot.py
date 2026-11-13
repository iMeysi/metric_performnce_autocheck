"""Static upper-tail POT, using vendored upstream SPOT initialization only."""
import numpy as np


def calibrate(scores, q=0.001, level=0.95, min_peaks=20, minimum=500):
    from omni_anomaly.spot import SPOT
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or len(s) < minimum or not np.isfinite(s).all():
        raise ValueError('insufficient/invalid calibration scores')
    if not 0 < q < 1-level < 1:
        raise ValueError('require 0 < q < 1-level < 1')
    initial = np.sort(s)[int(level*len(s))]  # same order statistic as SPOT
    peaks = s[s > initial] - initial
    if len(peaks) < min_peaks or np.ptp(peaks) <= 1e-12:
        raise ValueError('insufficient or degenerate POT tail; keep old model')
    spot = SPOT(q)
    spot.fit(s, np.array([], dtype=np.float64))
    spot.initialize(level=level, min_extrema=False, verbose=False)
    threshold = float(spot.extreme_quantile)
    if not np.isfinite(threshold) or threshold <= initial:
        raise ValueError('invalid POT fit; keep old model')
    return dict(threshold=threshold, q=q, level=level,
                initial_threshold=float(initial), peaks=int(len(peaks)),
                count=int(len(s)), method='static-SPOT-initialize-upper-tail')
