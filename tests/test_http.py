"""HTTP boundary integration, using synthetic Prometheus and a fake ML backend."""
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests

from pad_omni.data import Prometheus, collect_frame
from pad_omni.service import Metrics, ThreadingHTTPServer, make_http_server
from pad_omni.worker import train, infer
from test_core import config, FakeBackend


class SyntheticPrometheus(BaseHTTPRequestHandler):
    requests_seen = []
    broken = False

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != '/api/v1/query_range':
            self.send_error(404)
            return
        p = parse_qs(parsed.query)
        start, end, step = (int(p[k][0]) for k in ['start', 'end', 'step'])
        type(self).requests_seen.append((start, end, step))
        values = [[t, str(2 + np.sin(t/800.) + 0.1*np.cos(t/31.))]
                  for t in range(start, end+1, step)]
        if type(self).broken:
            values = []
        doc = dict(status='success', data=dict(resultType='matrix', result=[
            dict(metric={'job': 'api'}, values=values)]))
        body = json.dumps(doc).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def start_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


class HTTPIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.c = config()
        SyntheticPrometheus.broken = False
        SyntheticPrometheus.requests_seen = []
        self.prom = ThreadingHTTPServer(('127.0.0.1', 0), SyntheticPrometheus)
        self.prom_thread = start_server(self.prom)
        self.c['prometheus_url'] = 'http://127.0.0.1:%d' % self.prom.server_port
        self.metrics = Metrics(self.c)
        self.exporter = make_http_server(self.metrics, 0, '127.0.0.1')
        self.export_thread = start_server(self.exporter)
        self.url = 'http://127.0.0.1:%d' % self.exporter.server_port

    def tearDown(self):
        for server, thread in [(self.exporter, self.export_thread), (self.prom, self.prom_thread)]:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def get(self, path):
        return requests.get(self.url+path, timeout=5)

    def test_full_http_collection_train_restore_export_and_data_failure(self):
        self.assertEqual(self.get('/healthz').status_code, 200)
        self.assertEqual(self.get('/readyz').status_code, 503)
        self.assertEqual(self.get('/not-found').status_code, 404)
        self.assertNotIn('pad_omni_anomaly_score{', self.get('/metrics').text)
        g = self.c['groups'][0]
        with tempfile.TemporaryDirectory() as tmp:
            train(self.c, g, tmp, FakeBackend)
            result = infer(self.c, g, tmp, FakeBackend)
            self.metrics.update(g['name'], result=result)
            self.assertEqual(self.get('/readyz').status_code, 200)
            response = self.get('/metrics')
            self.assertIn('text/plain', response.headers['Content-Type'])
            self.assertIn('pad_omni_anomaly_score{group="api"}', response.text)
            self.assertIn('pad_omni_inference_peak_rss_bytes{group="api"}', response.text)
            self.assertTrue(all((e-s)//step+1 <= self.c['chunk_rows']
                                for s, e, step in SyntheticPrometheus.requests_seen))
            SyntheticPrometheus.broken = True
            with self.assertRaises(ValueError):
                infer(self.c, g, tmp, FakeBackend)
            self.metrics.fail(g['name'], 'inference')
            self.assertEqual(self.get('/readyz').status_code, 503)
            self.assertEqual(self.get('/healthz').status_code, 200)
            self.assertNotIn('pad_omni_anomaly_score{', self.get('/metrics').text)

    def test_real_http_dataframe_and_ttl(self):
        client = Prometheus(self.c)
        try:
            frame = collect_frame(client, ['a', 'b'], 0, 120, self.c)
        finally:
            client.close()
        self.assertEqual(frame.shape, (3, 2))
        self.assertEqual(str(frame.index.tz), 'UTC')
        now = time.time()
        self.metrics.update('api', result=dict(score=4, threshold=3, anomaly=1,
            completed_at=now-self.c['metric_ttl']-1, evaluated_at=now, model_trained_at=now))
        self.assertEqual(self.get('/readyz').status_code, 503)
        self.assertNotIn('pad_omni_anomaly_score{', self.get('/metrics').text)


if __name__ == '__main__':
    unittest.main()
