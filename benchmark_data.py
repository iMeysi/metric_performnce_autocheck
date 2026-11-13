"""Microbenchmark of data handling and real SPOT only; NOT ML performance."""
import argparse
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pad_omni.config import load
from pad_omni.data import clean
from pad_omni.preprocess import Standardizer, window_batches
from pad_omni.pot import calibrate
from pad_omni.worker import peak_rss_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, default=4321)
    p.add_argument('--features', type=int, default=3)
    args = p.parse_args()
    c = load(str(ROOT/'config.yaml'))
    if not 500 <= args.rows <= c['max_rows'] or not 1 <= args.features <= c['max_features']:
        p.error('rows/features outside configured benchmark budgets')
    started = time.monotonic()
    x = np.random.RandomState(42).normal(size=(args.rows, args.features))
    x = clean(x, c)
    x = Standardizer().fit(x[:int(len(x)*0.56)]).transform(x)
    n_windows = sum(len(b) for b in window_batches(x, c['window_length'], c['batch_size']))
    preprocessing = time.monotonic()-started
    started = time.monotonic()
    pot = calibrate(np.random.RandomState(7).exponential(size=3000))
    print(json.dumps(dict(scope='synthetic data preprocessing + real SPOT; NO model execution',
        python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, rows=args.rows, features=args.features,
        windows=n_windows, preprocessing_seconds=preprocessing,
        pot_seconds=time.monotonic()-started, peak_process_rss_bytes=peak_rss_bytes(),
        pot_threshold=pot['threshold']), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
