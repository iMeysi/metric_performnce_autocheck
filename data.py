"""Bounded query_range collector; each (query, label-set) is one feature."""
import json
import os
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def feature_key(query, labels):
    return json.dumps({'query': query, 'labels': labels}, sort_keys=True,
                      separators=(',', ':'))


class Prometheus:
    def __init__(self, config):
        self.c = config
        self.session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retry))
        self.session.mount('http://', HTTPAdapter(max_retries=retry))

    def query_range(self, query, start, end, step):
        token = os.getenv('FLT_PROM_ACCESS_TOKEN', '')
        token_file = os.getenv('PROM_TOKEN_FILE')
        if token_file:
            with open(token_file) as f:
                token = f.read().strip()  # reread rotated service account token
        headers = {'Authorization': 'Bearer ' + token} if token else {}
        verify = os.getenv('PROM_CA_FILE') or True
        params = dict(query=query, start=start, end=end, step=step,
                      timeout=str(self.c['request_timeout']) + 's')
        deadline = time.monotonic() + self.c['request_timeout'] * 4
        with self.session.get(self.c['prometheus_url'].rstrip('/') + '/api/v1/query_range',
                              params=params, headers=headers, verify=verify,
                              timeout=(5, self.c['request_timeout']), stream=True) as r:
            r.raise_for_status()
            body = bytearray()
            for block in r.iter_content(65536):
                if len(body) + len(block) > self.c['max_response_bytes']:
                    raise ValueError('Prometheus response exceeds byte budget')
                if time.monotonic() > deadline:
                    raise TimeoutError('response deadline exceeded')
                body.extend(block)
            doc = json.loads(body)
        if doc.get('status') != 'success' or doc.get('warnings'):
            raise ValueError('Prometheus error or partial/warning response')
        if doc['data']['resultType'] != 'matrix':
            raise ValueError('query_range must return a matrix')
        return doc['data']['result']

    def close(self):
        self.session.close()


def collect(client, queries, start, end, c, expected=None):
    step = c['step_seconds']
    if start % step or end % step or end < start:
        raise ValueError('bounds must be on the common epoch grid')
    row_count = (end-start)//step + 1
    if row_count > c['max_rows']:
        raise ValueError('row budget exceeded')
    grid = np.arange(start, end + 1, step, dtype=np.int64)
    columns = {}
    allowed = set(expected) if expected is not None else None
    discovered = set()
    for query in queries:
        for offset in range(0, len(grid), c['chunk_rows']):
            stop = min(offset + c['chunk_rows'], len(grid))
            rows = client.query_range(query, int(grid[offset]), int(grid[stop-1]), step)
            chunk_seen = set()
            for series in rows:
                key = feature_key(query, series['metric'])
                if key in chunk_seen:
                    raise ValueError('duplicate label-set in one response')
                chunk_seen.add(key)
                discovered.add(key)
                if len(discovered) > c['max_features']:
                    raise ValueError('feature cardinality budget exceeded; aggregate PromQL')
                if allowed is not None and key not in allowed:
                    continue  # additions never silently change trained input dimensions
                if key not in columns:
                    columns[key] = np.full(len(grid), np.nan, dtype=np.float64)
                seen_ts = set()
                for timestamp, value in series.get('values', []):
                    ts = float(timestamp)
                    i = int(round((ts-start)/step))
                    if i < offset or i >= stop or abs(ts - grid[i]) > 0.001:
                        raise ValueError('off-grid or out-of-chunk timestamp')
                    if i in seen_ts:
                        raise ValueError('duplicate timestamp')
                    seen_ts.add(i)
                    x = float(value)
                    columns[key][i] = x if np.isfinite(x) else np.nan
    features = list(expected) if expected is not None else sorted(columns)
    if not features or len(features) > c['max_features']:
        raise ValueError('empty/oversized feature schema')
    matrix = np.column_stack([columns.get(k, np.full(len(grid), np.nan)) for k in features])
    return grid, matrix, features


def collect_frame(client, queries, start, end, c, expected=None):
    """Aligned UTC DataFrame; canonical JSON feature keys preserve label identity."""
    grid, values, features = collect(client, queries, start, end, c, expected)
    return pd.DataFrame(values, index=pd.to_datetime(grid, unit='s', utc=True),
                        columns=features)


def clean(matrix, c, online=False):
    x = np.asarray(matrix, dtype=np.float64).copy()
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError('nonempty 2-D matrix required')
    x[~np.isfinite(x)] = np.nan
    missing = np.isnan(x)
    if online and missing[-1].any():
        raise ValueError('latest evaluation point missing; do not export stale inference')
    if missing.mean() > c['max_missing_fraction']:
        raise ValueError('missing-data budget exceeded')
    if c['fill_limit']:
        x = pd.DataFrame(x).ffill(limit=c['fill_limit']).values
    # Never bfill/interpolate using future values, drop rows, or substitute zeros.
    if not np.isfinite(x).all():
        raise ValueError('unresolved gaps; need a contiguous complete time grid')
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('data exceeds float32 range')
    return x
