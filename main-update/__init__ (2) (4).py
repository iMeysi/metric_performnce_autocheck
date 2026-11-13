"""Run the REAL TF1 model against a local synthetic Prometheus HTTP API.
Run inside the legacy image: python scripts/smoke_real.py --epochs 1
No FakeBackend is injected. Fails on training, POT or fresh-process restore errors.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
import yaml
from test_http import SyntheticPrometheus, start_server
from pad_omni.config import load
from pad_omni.service import ThreadingHTTPServer, invoke


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=1)
    args = p.parse_args()
    # The fixture must never contact a production endpoint or inherit its token.
    for key in list(os.environ):
        if key.startswith('FLT_') or key in ['PROM_TOKEN_FILE', 'PROM_CA_FILE']:
            del os.environ[key]
    c = load(str(ROOT/'config.yaml'))
    c['max_epoch'] = args.epochs
    server = ThreadingHTTPServer(('127.0.0.1', 0), SyntheticPrometheus)
    thread = start_server(server)
    try:
        with tempfile.TemporaryDirectory(prefix='omni-real-smoke-') as tmp:
            tmp = Path(tmp)
            c['prometheus_url'] = 'http://127.0.0.1:%d' % server.server_port
            c['model_dir'] = str(tmp/'models')
            path = tmp/'config.yaml'
            with open(path, 'w') as f:
                yaml.safe_dump(c, f)
            directory = tmp/'model'
            result = tmp/'result.json'
            invoke('train', path, c['groups'][0]['name'], directory, c['train_timeout'])
            invoke('infer', path, c['groups'][0]['name'], directory, c['inference_timeout'], result)
            report = json.loads(result.read_text())
            manifest = json.loads((directory/'manifest.json').read_text())
            report.update(training_seconds=manifest['training_seconds'],
                training_peak_rss_bytes=manifest['training_peak_rss_bytes'],
                backend='REAL OmniAnomaly / TensorFlow 1.12',
                data='synthetic HTTP; not an accuracy or capacity validation')
            print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()
