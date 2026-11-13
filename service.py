"""PAD-compatible companion exporter; TF is never imported by this process."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily

from .config import load
from .worker import read_manifest, write_json

LOG = logging.getLogger(__name__)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Metrics:
    def __init__(self, c):
        self.c = c
        self.lock = threading.Lock()
        self.states = {g['name']: dict(result=None, training=False,
                    inference_errors=0, training_errors=0, training_seconds=None,
                    training_peak_rss_bytes=None) for g in c['groups']}

    def update(self, name, **fields):
        with self.lock:
            self.states[name].update(fields)

    def fail(self, name, kind):
        with self.lock:
            self.states[name][kind+'_errors'] += 1
            if kind == 'inference':
                self.states[name]['result'] = None

    def fresh(self, r, now):
        try:
            return bool(r and all(math.isfinite(r[k]) for k in
                ['score', 'threshold', 'completed_at', 'evaluated_at', 'model_trained_at'])
                and r['anomaly'] == int(r['score'] > r['threshold'])
                and 0 <= now-r['completed_at'] <= self.c['metric_ttl']
                and 0 <= now-r['evaluated_at'] <= self.c['lag_seconds'] +
                    self.c['step_seconds'] + self.c['metric_ttl']
                and 0 <= now-r['model_trained_at'] <= self.c['max_model_age'])
        except (KeyError, TypeError, ValueError):
            return False

    def ready(self):
        with self.lock:
            return all(self.fresh(s['result'], time.time()) for s in self.states.values())

    def collect(self):
        definitions = {
            'ready': 'Fresh valid inference exists (not the anomaly status)',
            'training': 'Retraining queued or running',
            'anomaly_score': 'Negative joint reconstruction log probability, higher is worse',
            'anomaly_threshold': 'Static upper-tail POT threshold',
            'anomaly': 'One when score is strictly greater than threshold',
            'last_evaluation_timestamp_seconds': 'Prometheus evaluation timestamp',
            'model_trained_timestamp_seconds': 'Model completion timestamp',
            'training_seconds': 'Last promoted model training duration including collection and POT',
            'training_peak_rss_bytes': 'Peak RSS of the last promoted training worker (not whole pod)',
            'inference_seconds': 'Last successful inference duration including collection and graph restore',
            'inference_peak_rss_bytes': 'Peak RSS of last successful inference worker (not whole pod)',
        }
        gauges = {k: GaugeMetricFamily('pad_omni_'+k, v, labels=['group'])
                  for k, v in definitions.items()}
        errors = {k: CounterMetricFamily('pad_omni_'+k+'_errors',
            'Failed '+k+' operations', labels=['group']) for k in ['training', 'inference']}
        fields = dict(anomaly_score='score', anomaly_threshold='threshold',
            anomaly='anomaly', last_evaluation_timestamp_seconds='evaluated_at',
            model_trained_timestamp_seconds='model_trained_at')
        now = time.time()
        with self.lock:
            for name, s in self.states.items():
                ok = self.fresh(s['result'], now)
                gauges['ready'].add_metric([name], int(ok))
                gauges['training'].add_metric([name], int(s['training']))
                for k in ['training_seconds', 'training_peak_rss_bytes']:
                    if s[k] is not None:
                        gauges[k].add_metric([name], s[k])
                for k, family in errors.items():
                    family.add_metric([name], s[k+'_errors'])
                if ok:
                    for k in ['inference_seconds', 'inference_peak_rss_bytes']:
                        if k in s['result']:
                            gauges[k].add_metric([name], s['result'][k])
                    for k, field in fields.items():
                        gauges[k].add_metric([name], s['result'][field])
        yield from gauges.values()
        yield from errors.values()


def invoke(mode, config_path, group, directory, timeout, result=None):
    command = [sys.executable, '-m', 'pad_omni.worker', mode,
        '--config', str(config_path), '--group', group, '--directory', str(directory)]
    if result:
        command += ['--result', str(result)]
    # Worker has no child processes. run() kills and waits for it on timeout.
    subprocess.run(command, timeout=timeout, check=True, stdout=subprocess.DEVNULL)


def train_candidate(c, config_path, group, root):
    directory = Path(tempfile.mkdtemp(prefix='v-', dir=str(root)))
    try:
        invoke('train', config_path, group['name'], directory, c['train_timeout'])
        read_manifest(directory, c, group)
        # Validate checkpoint restore in a fresh process before promotion.
        with tempfile.TemporaryDirectory(prefix='verify-') as tmp:
            result_path = Path(tmp)/'result.json'
            invoke('infer', config_path, group['name'], directory,
                   c['inference_timeout'], result_path)
            with open(result_path) as f:
                result = json.load(f)
            if not Metrics(c).fresh(result, time.time()):
                raise ValueError('candidate inference is stale')
        return directory
    except BaseException:
        shutil.rmtree(str(directory), ignore_errors=True)
        raise


def group_loop(c, config_path, group, metrics, stop, trainers, inference_slots):
    name = group['name']
    root = Path(c['model_dir']) / name
    root.mkdir(parents=True, exist_ok=True)
    current = None
    try:
        with open(root/'CURRENT') as f:
            version = json.load(f)['version']
        if not version.startswith('v-') or Path(version).name != version:
            raise ValueError('unsafe version pointer')
        current = root/version
        read_manifest(current, c, group)
    except (OSError, ValueError, KeyError):
        current = None
    # Exclusive service lock makes leftover candidates from a crash safe to remove.
    for entry in root.glob('v-*'):
        if entry != current:
            shutil.rmtree(str(entry), ignore_errors=True)
    future, next_training = None, 0
    if current:
        manifest = read_manifest(current, c, group)
        metrics.update(name, **{k: manifest.get(k) for k in
                       ['training_seconds', 'training_peak_rss_bytes']})
        next_training = manifest['trained_at'] + c['retrain_seconds']
    while not stop.is_set():
        tick = time.monotonic()
        if future is not None and future.done():
            try:
                candidate = future.result()
                manifest = read_manifest(candidate, c, group)
                old = current
                write_json(root/'CURRENT', dict(version=candidate.name))
                current = candidate
                metrics.update(name, **{k: manifest.get(k) for k in
                               ['training_seconds', 'training_peak_rss_bytes']})
                if old and old != current:
                    shutil.rmtree(str(old), ignore_errors=True)
                LOG.info('model promoted group=%s version=%s', name, current.name)
            except Exception:
                metrics.fail(name, 'training')
                LOG.exception('training failed; retain old model group=%s', name)
            future = None
            metrics.update(name, training=False)
            next_training = time.time()+c['retrain_seconds']
        if future is None and time.time() >= next_training:
            metrics.update(name, training=True)
            future = trainers.submit(train_candidate, c, config_path, group, root)
        if current:
            try:
                read_manifest(current, c, group)
                with inference_slots:
                    if stop.is_set():
                        break
                    with tempfile.TemporaryDirectory(prefix='infer-') as tmp:
                        result_path = Path(tmp)/'result.json'
                        invoke('infer', config_path, name, current,
                               c['inference_timeout'], result_path)
                        with open(result_path) as f:
                            result = json.load(f)
                if not metrics.fresh(result, time.time()):
                    raise ValueError('inference completed too late')
                metrics.update(name, result=result)
            except Exception:
                metrics.fail(name, 'inference')
                LOG.exception('inference unavailable group=%s', name)
        else:
            metrics.update(name, result=None)
        stop.wait(max(0.1, c['poll_seconds']-(time.monotonic()-tick)))


def make_http_server(metrics, port, host='0.0.0.0'):
    registry = CollectorRegistry()
    registry.register(metrics)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/metrics':
                body, status, content_type = generate_latest(registry), 200, CONTENT_TYPE_LATEST
            elif self.path in ['/healthz', '/readyz']:
                ok = self.path == '/healthz' or metrics.ready()
                body, status, content_type = (b'ok\n' if ok else b'not ready\n'), (200 if ok else 503), 'text/plain'
            else:
                body, status, content_type = b'not found\n', 404, 'text/plain'
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(config_path):
    c = load(config_path)
    root = Path(c['model_dir'])
    root.mkdir(parents=True, exist_ok=True)
    # Only one writer per PVC/model directory. Keep descriptor open for lifetime.
    lock = open(root/'.service.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metrics = Metrics(c)
    stop = threading.Event()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stop.set())

    server = make_http_server(metrics, c['port'])
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    slots = threading.BoundedSemaphore(c['parallelism'])
    with ThreadPoolExecutor(max_workers=c['parallelism']) as trainers:
        threads = [threading.Thread(target=group_loop,
            args=(c, config_path, g, metrics, stop, trainers, slots), daemon=True)
            for g in c['groups']]
        for thread in threads:
            thread.start()
        while not stop.wait(1):
            if not all(t.is_alive() for t in threads):
                stop.set()
                LOG.error('group thread died; terminating for supervisor restart')
        server.shutdown()
        server.server_close()
        for thread in threads:
            thread.join()
    lock.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='/app/config.yaml')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    serve(str(Path(args.config).resolve()))


if __name__ == '__main__':
    main()
