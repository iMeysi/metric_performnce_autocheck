import hashlib
import json
import os
import re

import yaml


def duration(value):
    m = re.fullmatch(r"([1-9][0-9]*)(s|m|h|d)", str(value))
    if not m:
        raise ValueError('duration must be a positive integer with s/m/h/d')
    return int(m[1]) * {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}[m[2]]


def load(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    if not isinstance(c, dict):
        raise ValueError('configuration must be a mapping')
    defaults = dict(step_seconds=60, poll_seconds=60, lag_seconds=120,
                    rolling_seconds=259200, retrain_seconds=7200,
                    parallelism=1, max_groups=8, max_features=32, max_rows=10000,
                    chunk_rows=1000, max_response_bytes=16777216,
                    fill_limit=2, max_missing_fraction=0.02, window_length=100,
                    batch_size=32, max_epoch=10, hidden=128, dense=128,
                    z_dim=3, nf_layers=20, n_z=8, seed=42, tf_threads=1,
                    pot_q=0.001, pot_level=0.95, min_peaks=20,
                    min_calibration=500, train_timeout=3600,
                    request_timeout=30, inference_timeout=120, metric_ttl=240,
                    max_model_age=86400, port=8080, model_dir='/data/models')
    for k, v in defaults.items():
        c.setdefault(k, v)
    c['prometheus_url'] = os.getenv('FLT_PROM_URL', c.get('prometheus_url', ''))
    if os.getenv('FLT_METRICS_LIST'):
        c['groups'] = [dict(name='default', queries=[q.strip() for q in
                      os.environ['FLT_METRICS_LIST'].split(';') if q.strip()])]
    if os.getenv('FLT_ROLLING_TRAINING_WINDOW_SIZE'):
        c['rolling_seconds'] = duration(os.environ['FLT_ROLLING_TRAINING_WINDOW_SIZE'])
    if os.getenv('FLT_RETRAINING_INTERVAL_MINUTES'):
        c['retrain_seconds'] = int(os.environ['FLT_RETRAINING_INTERVAL_MINUTES']) * 60
    if os.getenv('FLT_PARALLELISM'):
        c['parallelism'] = int(os.environ['FLT_PARALLELISM'])
    if not c['prometheus_url'].startswith(('http://', 'https://')):
        raise ValueError('prometheus_url / FLT_PROM_URL required')
    groups = c.get('groups', [])
    if not 1 <= len(groups) <= c['max_groups']:
        raise ValueError('invalid number of groups')
    names = set()
    for g in groups:
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', g['name']) or g['name'] in names:
            raise ValueError('group names must be unique and path-safe')
        names.add(g['name'])
        if not g.get('queries') or any(not isinstance(q, str) or not q.strip() for q in g['queries']):
            raise ValueError('nonempty queries required')
        if len(g['queries']) > c['max_features'] or len(set(g['queries'])) != len(g['queries']):
            raise ValueError('too many or duplicate queries')
    for k in defaults:
        if isinstance(defaults[k], int) and k not in ('lag_seconds', 'fill_limit', 'seed'):
            if not isinstance(c[k], int) or c[k] < 1:
                raise ValueError(k + ' must be a positive integer')
    if c['fill_limit'] < 0 or c['lag_seconds'] < 0:
        raise ValueError('negative fill/lag')
    if not 0 <= c['max_missing_fraction'] < 1:
        raise ValueError('invalid missing fraction')
    if not 0 < c['pot_q'] < 1-c['pot_level'] < 1:
        raise ValueError('require 0 < q < 1 - level < 1')
    if c['rolling_seconds'] // c['step_seconds'] + 1 > c['max_rows']:
        raise ValueError('rolling history exceeds max_rows')
    if c['window_length'] < 2 or c['port'] > 65535:
        raise ValueError('invalid window or port')
    if c['max_model_age'] <= c['retrain_seconds']:
        raise ValueError('max_model_age must exceed retrain_seconds')
    if c['metric_ttl'] < c['poll_seconds']:
        raise ValueError('metric_ttl must cover poll_seconds')
    c['parallelism'] = min(c['parallelism'], len(groups), os.cpu_count() or 1)
    return c


def signature(c, group):
    keys = ['step_seconds', 'window_length', 'hidden', 'dense', 'z_dim',
            'nf_layers', 'n_z', 'fill_limit', 'pot_q', 'pot_level']
    payload = {'group': group, 'settings': {k: c[k] for k in keys}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
