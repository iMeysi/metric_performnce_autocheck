"""One train or inference operation per subprocess; no label-aware evaluation."""
import argparse
import json
import os
import resource
from pathlib import Path
import sys
import time

import numpy as np

VENDOR = Path(__file__).resolve().parents[1] / 'vendor' / 'OmniAnomaly'
sys.path.insert(0, str(VENDOR))

from .config import load, signature
from .data import Prometheus, collect_frame, clean
from .preprocess import Standardizer


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def write_json(path, value):
    with open(str(path) + '.tmp', 'w') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(path) + '.tmp', str(path))


def read_manifest(directory, c, group, now=None):
    with open(Path(directory) / 'manifest.json') as f:
        m = json.load(f)
    now = time.time() if now is None else now
    if m['format'] != 1 or m['signature'] != signature(c, group):
        raise ValueError('artifact/config signature mismatch; retrain required')
    if not 0 <= now-m['trained_at'] <= c['max_model_age']:
        raise ValueError('model expired or future-dated')
    if (not 1 <= len(m['features']) <= c['max_features'] or
            len(set(m['features'])) != len(m['features']) or
            not np.isfinite(m['pot']['threshold'])):
        raise ValueError('invalid artifact schema/threshold')
    return m


def train(c, group, output, backend_class=None):
    started = time.monotonic()
    from .pot import calibrate
    if backend_class is None:
        from .backend import Backend
        backend_class = Backend
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    end = int((time.time()-c['lag_seconds']) // c['step_seconds']) * c['step_seconds']
    start = end - (c['rolling_seconds']//c['step_seconds']) * c['step_seconds']
    client = Prometheus(c)
    try:
        frame = collect_frame(client, group['queries'], start, end, c)
        raw, features = frame.values, list(frame.columns)
    finally:
        client.close()
    # Outer chronological 70/30 split: Trainer uses its own 80/20 split
    # inside the first 70%; calibration is entirely outside Trainer.fit().
    cut = int(len(raw)*0.7)
    valid = int(cut*0.2)
    fit_count = cut-valid
    w = c['window_length']
    if (fit_count-w+1 < c['batch_size'] or valid < w or
            len(raw)-cut-w+1 < c['min_calibration']):
        raise ValueError('history too short for train/validation/calibration split')
    values = clean(raw, c)
    scaler = Standardizer().fit(values[:fit_count])
    z = scaler.transform(values)
    backend = backend_class(c, len(features))
    try:
        backend.fit(z[:cut])
        scores = backend.scores(z[cut:])
        pot = calibrate(scores, c['pot_q'], c['pot_level'],
                        c['min_peaks'], c['min_calibration'])
        backend.save(output / 'model')
    finally:
        backend.close()
    scaler.save(output / 'scaler.npz')
    write_json(output / 'manifest.json', dict(format=1,
        signature=signature(c, group), group=group['name'], features=features,
        trained_at=time.time(), history_start=start, history_end=end,
        training_seconds=time.monotonic()-started, training_peak_rss_bytes=peak_rss_bytes(),
        training_rows=fit_count, validation_rows=valid,
        calibration_rows=len(raw)-cut, window_length=w,
        score='negative joint reconstruction log probability; last point', pot=pot))


def infer(c, group, directory, backend_class=None):
    started = time.monotonic()
    if backend_class is None:
        from .backend import Backend
        backend_class = Backend
    m = read_manifest(directory, c, group)
    end = int((time.time()-c['lag_seconds']) // c['step_seconds']) * c['step_seconds']
    # Extra past context allows strictly forward-only filling before the window.
    rows = c['window_length'] + c['fill_limit']
    start = end-(rows-1)*c['step_seconds']
    client = Prometheus(c)
    try:
        raw = collect_frame(client, group['queries'], start, end, c, m['features']).values
    finally:
        client.close()
    z = Standardizer.load(Path(directory)/'scaler.npz').transform(clean(raw, c, online=True))
    backend = backend_class(c, len(m['features']))
    try:
        backend.restore(Path(directory)/'model')
        score = float(backend.scores(z[-c['window_length']:])[-1])
    finally:
        backend.close()
    if not np.isfinite(score):
        raise ValueError('nonfinite inference score')
    return dict(score=score, threshold=m['pot']['threshold'],
        anomaly=int(score > m['pot']['threshold']), evaluated_at=end,
        completed_at=time.time(), model_trained_at=m['trained_at'],
        inference_seconds=time.monotonic()-started, inference_peak_rss_bytes=peak_rss_bytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'infer'])
    p.add_argument('--config', required=True)
    p.add_argument('--group', required=True)
    p.add_argument('--directory', required=True)
    p.add_argument('--result')
    args = p.parse_args()
    c = load(args.config)
    group = next(g for g in c['groups'] if g['name'] == args.group)
    if args.mode == 'train':
        train(c, group, args.directory)
    else:
        if not args.result:
            p.error('--result required for inference')
        write_json(args.result, infer(c, group, args.directory))


if __name__ == '__main__':
    main()
