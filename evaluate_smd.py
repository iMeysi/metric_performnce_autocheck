"""Offline pointwise evaluation. Place this file in the modern project root."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor' / 'OmniAnomaly'))
from pad_omni.config import load
from pad_omni.backend import Backend
from pad_omni.preprocess import Standardizer
from pad_omni.pot import calibrate


def main():
    data = ROOT / 'data' / 'smd'
    def matrix(part):
        return np.loadtxt(data / part / 'machine-1-1.txt', delimiter=',', dtype=np.float32, ndmin=2)
    train, test = matrix('train'), matrix('test')
    if train.shape[1] != test.shape[1] or not all(np.isfinite(x).all() for x in (train, test)):
        raise ValueError('Invalid feature matrices')
    c = load(str(ROOT / 'config.windows.yaml'))
    c['max_epoch'] = 1
    cut = int(0.7 * len(train))
    scaler = Standardizer().fit(train[:cut-int(0.2*cut)])
    model = Backend(c, train.shape[1])
    try:
        print('Training on machine-1-1 (CPU) ...', flush=True)
        model.fit(scaler.transform(train[:cut]))
        print('Calibrating threshold on held-out training data ...', flush=True)
        th = calibrate(model.scores(scaler.transform(train[cut:])), c['pot_q'], c['pot_level'], c['min_peaks'], c['min_calibration'])['threshold']
        print('Scoring test windows ...', flush=True)
        scores = model.scores(scaler.transform(test))
    finally:
        model.close()
    labels = matrix('test_label')
    if labels.shape != (len(test), 1) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid labels')
    truth = labels[c['window_length']-1:, 0].astype(bool)
    if scores.shape != truth.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid or misaligned scores')
    pred = scores > th
    tp, fp, fn = (int(x.sum()) for x in (truth & pred, ~truth & pred, truth & ~pred))
    def ratio(a, b): return a / b if b else None
    report = dict(machine='machine-1-1', epochs=1, threshold=th, evaluated_points=len(scores), tp=tp, fp=fp, fn=fn, precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn), f1=ratio(2*tp,2*tp+fp+fn))
    text = json.dumps(report, indent=2, allow_nan=False)
    (ROOT / 'smd-result.json').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
