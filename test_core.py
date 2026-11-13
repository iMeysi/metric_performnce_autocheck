"""Fast contract tests. No TensorFlow import; run with unittest discovery."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

import numpy as np
from prometheus_client import CollectorRegistry, generate_latest

from pad_omni.config import duration, load, signature
from pad_omni.data import collect, clean, feature_key, Prometheus
from pad_omni.preprocess import Standardizer, window_batches
from pad_omni.pot import calibrate
from pad_omni.worker import train, infer, read_manifest, write_json
from pad_omni.service import Metrics, train_candidate, invoke

ROOT = Path(__file__).resolve().parents[1]


def config():
    with patch.dict('os.environ', {}, clear=True):
        return load(str(ROOT/'config.yaml'))


class FakePrometheus:
    def __init__(self, c=None):
        self.c = c

    def close(self):
        pass

    def query_range(self, query, start, end, step):
        return [{'metric': {'job': 'api'}, 'values': [
            [t, str(2+np.sin(t/800.0))] for t in range(start, end+1, step)]}]


class FakeBackend:
    """Tests orchestration only; not a substitute for the real model."""
    def __init__(self, c, dimensions):
        self.c, self.dimensions = c, dimensions

    def fit(self, x):
        assert x.ndim == 2 and x.shape[1] == self.dimensions

    def scores(self, x):
        return np.random.RandomState(5).exponential(size=len(x)-self.c['window_length']+1)

    def save(self, path):
        Path(str(path)+'.index').write_text('FAKE TEST ARTIFACT')

    def restore(self, path):
        assert Path(str(path)+'.index').exists()

    def close(self):
        pass


class ConfigTests(unittest.TestCase):
    def test_duration(self):
        self.assertEqual(duration('15d'), 1296000)
        for value in ['0m', '-2h', '1.5d', '15']:
            with self.assertRaises(ValueError):
                duration(value)

    def test_environment_and_query_delimiter(self):
        with patch.dict('os.environ', {'FLT_PROM_URL': 'https://example.test',
            'FLT_METRICS_LIST': 'sum(foo{a="x",b="y"});sum(bar)',
            'FLT_ROLLING_TRAINING_WINDOW_SIZE': '2d',
            'FLT_RETRAINING_INTERVAL_MINUTES': '15', 'FLT_PARALLELISM': '2'}, clear=True):
            c = load(str(ROOT/'config.yaml'))
        self.assertEqual(len(c['groups'][0]['queries']), 2)
        self.assertEqual(c['rolling_seconds'], 172800)
        self.assertEqual(c['retrain_seconds'], 900)

    def test_rolling_bound(self):
        with patch.dict('os.environ', {'FLT_ROLLING_TRAINING_WINDOW_SIZE': '15d'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'exceeds max_rows'):
                load(str(ROOT/'config.yaml'))

    def test_signature_order_matters(self):
        c = config()
        group = c['groups'][0]
        reversed_group = dict(group, queries=list(reversed(group['queries'])))
        self.assertNotEqual(signature(c, group), signature(c, reversed_group))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.c = config()
        self.c.update(chunk_rows=2, max_missing_fraction=0.5)

    def test_chunked_alignment_and_schema(self):
        grid, x, features = collect(FakePrometheus(), ['a', 'b'], 0, 240, self.c)
        self.assertEqual(x.shape, (5, 2))
        self.assertEqual(grid.tolist(), [0, 60, 120, 180, 240])
        _, y, order = collect(FakePrometheus(), ['a', 'b'], 0, 240, self.c, features[::-1])
        np.testing.assert_equal(x[:, ::-1], y)
        self.assertEqual(order, features[::-1])

    def test_missing_feature_is_not_zero(self):
        _, x, _ = collect(FakePrometheus(), ['a'], 0, 120, self.c, ['absent'])
        self.assertTrue(np.isnan(x).all())
        with self.assertRaises(ValueError):
            clean(x, self.c)

    def test_extra_features_do_not_change_schema(self):
        expected = [feature_key('a', {'job': 'api'})]
        _, x, features = collect(FakePrometheus(), ['a', 'b'], 0, 120, self.c, expected)
        self.assertEqual(features, expected)
        self.assertEqual(x.shape, (3, 1))

    def test_feature_and_row_limits(self):
        self.c['max_features'] = 1
        with self.assertRaisesRegex(ValueError, 'cardinality'):
            collect(FakePrometheus(), ['a', 'b'], 0, 120, self.c)
        self.c['max_rows'] = 2
        with self.assertRaisesRegex(ValueError, 'row budget'):
            collect(FakePrometheus(), ['a'], 0, 120, self.c)

    def test_bad_timestamps_and_duplicate_series(self):
        fake = Mock()
        for rows in [
            [{'metric': {}, 'values': [[1, '1']]}],
            [{'metric': {}, 'values': [[0, '1'], [0, '2']]}],
            [{'metric': {}, 'values': []}, {'metric': {}, 'values': []}],
        ]:
            fake.query_range.return_value = rows
            with self.assertRaises(ValueError):
                collect(fake, ['a'], 0, 60, self.c)

    def test_forward_fill_only(self):
        x = clean([[1.], [np.nan], [3.]], self.c)
        self.assertEqual(x[:, 0].tolist(), [1., 1., 3.])
        with self.assertRaisesRegex(ValueError, 'unresolved gaps'):
            clean([[np.nan], [2.], [3.]], self.c)

    def test_missing_latest_online_rejected(self):
        with self.assertRaisesRegex(ValueError, 'latest evaluation'):
            clean([[1.], [2.], [np.nan]], self.c, online=True)

    def test_missing_budget(self):
        self.c['max_missing_fraction'] = 0.01
        with self.assertRaisesRegex(ValueError, 'budget'):
            clean([[1.], [np.nan], [3.]], self.c)

    def test_invalid_shape_and_overflow_rejected(self):
        for values in [[], [1., 2.], [[1e100]]]:
            with np.errstate(over='ignore'):
                with self.assertRaises(ValueError):
                    clean(values, self.c)

    def test_http_warning_matrix_and_size_checks(self):
        client = Prometheus(self.c)
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        client.session.get = Mock(return_value=response)
        for doc in [dict(status='error'), dict(status='success', warnings=['partial']),
                    dict(status='success', data=dict(resultType='vector'))]:
            response.iter_content.return_value = [json.dumps(doc).encode()]
            with self.assertRaises(ValueError):
                client.query_range('a', 0, 60, 60)
        self.c['max_response_bytes'] = 2
        response.iter_content.return_value = [b'xxxx']
        with self.assertRaisesRegex(ValueError, 'byte budget'):
            client.query_range('a', 0, 60, 60)
        client.close()


class PreprocessTests(unittest.TestCase):
    def test_scaler_roundtrip_and_constant_column(self):
        train_x = np.array([[1., 5.], [3., 5.]])
        scaler = Standardizer().fit(train_x)
        np.testing.assert_equal(scaler.transform(train_x)[:, 1], [0., 0.])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'s.npz'
            scaler.save(path)
            np.testing.assert_equal(Standardizer.load(path).transform(train_x), scaler.transform(train_x))
        self.assertGreater(scaler.transform([[100., 5.]])[0, 0], 1.)

    def test_window_alignment(self):
        x = np.arange(20).reshape(10, 2)
        batches = list(window_batches(x, 3, 4))
        windows = np.concatenate(batches)
        self.assertEqual(windows.shape, (8, 3, 2))
        np.testing.assert_equal(windows[:, -1], x[2:])

    def test_scaler_dimension_and_nan(self):
        scaler = Standardizer().fit([[1., 2.], [2., 3.]])
        for x in [[[1.]], [[np.nan, 1.]]]:
            with self.assertRaises(ValueError):
                scaler.transform(x)


class POTTests(unittest.TestCase):
    def test_actual_vendored_spot(self):
        scores = np.random.RandomState(7).exponential(size=3000)
        result = calibrate(scores)
        self.assertGreater(result['threshold'], result['initial_threshold'])
        self.assertGreaterEqual(result['peaks'], 20)

    def test_invalid_or_degenerate_tail(self):
        for x in [np.ones(1000), np.arange(10), np.full(1000, np.nan)]:
            with self.assertRaises(ValueError):
                calibrate(x)


class LifecycleTests(unittest.TestCase):
    @patch('pad_omni.worker.Prometheus', FakePrometheus)
    def test_train_persist_restore_infer_contract(self):
        c = config()
        g = c['groups'][0]
        with tempfile.TemporaryDirectory() as tmp:
            train(c, g, tmp, FakeBackend)
            m = read_manifest(tmp, c, g)
            self.assertEqual(len(m['features']), 3)
            self.assertGreater(m['pot']['count'], c['min_calibration'])
            result = infer(c, g, tmp, FakeBackend)
            self.assertEqual(result['anomaly'], int(result['score'] > result['threshold']))
            with self.assertRaisesRegex(ValueError, 'expired'):
                read_manifest(tmp, c, g, m['trained_at']+c['max_model_age']+1)
            changed = dict(c, window_length=50)
            with self.assertRaisesRegex(ValueError, 'signature'):
                read_manifest(tmp, changed, g)

    def test_failed_training_removes_candidate_not_previous(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp)/'v-old'
            old.mkdir()
            with patch('pad_omni.service.invoke', side_effect=TimeoutError):
                with self.assertRaises(TimeoutError):
                    train_candidate(c, 'config.yaml', c['groups'][0], Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [old])

    @patch('pad_omni.worker.Prometheus', FakePrometheus)
    def test_candidate_restore_verified_before_promotion(self):
        c = config()
        g = c['groups'][0]
        calls = []

        def fake_invoke(mode, config_path, group, directory, timeout, result=None):
            calls.append(mode)
            if mode == 'train':
                train(c, g, directory, FakeBackend)
            else:
                write_json(result, infer(c, g, directory, FakeBackend))

        with tempfile.TemporaryDirectory() as tmp:
            with patch('pad_omni.service.invoke', side_effect=fake_invoke):
                directory = train_candidate(c, 'unused', g, Path(tmp))
                self.assertTrue((directory/'manifest.json').exists())
                self.assertEqual(calls, ['train', 'infer'])
            old = Path(tmp)/'v-old'
            old.mkdir()
            def broken_restore(*args, **kwargs):
                if args[0] == 'infer':
                    raise RuntimeError('restore failed')
                return fake_invoke(*args, **kwargs)
            with patch('pad_omni.service.invoke', side_effect=broken_restore):
                with self.assertRaisesRegex(RuntimeError, 'restore failed'):
                    train_candidate(c, 'unused', g, Path(tmp))
            self.assertEqual(set(Path(tmp).iterdir()), {old, directory})

    def test_subprocess_timeout_is_passed(self):
        with patch('pad_omni.service.subprocess.run') as run:
            invoke('train', 'c.yaml', 'api', '/tmp/model', 9)
            self.assertEqual(run.call_args[1]['timeout'], 9)
            self.assertTrue(run.call_args[1]['check'])

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'CURRENT'
            write_json(path, dict(version='v-a'))
            self.assertEqual(json.loads(path.read_text())['version'], 'v-a')
            self.assertFalse(Path(str(path)+'.tmp').exists())


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.c = config()
        self.m = Metrics(self.c)
        self.registry = CollectorRegistry()
        self.registry.register(self.m)

    def test_fail_closed_and_staleness(self):
        text = generate_latest(self.registry).decode()
        self.assertNotIn('pad_omni_anomaly_score{', text)
        self.assertFalse(self.m.ready())
        now = time.time()
        r = dict(score=4., threshold=3., anomaly=1, completed_at=now,
                 evaluated_at=now-120, model_trained_at=now-10)
        self.m.update('api', result=r)
        self.assertTrue(self.m.ready())
        self.assertIn('pad_omni_anomaly{group="api"} 1.0', generate_latest(self.registry).decode())
        self.m.update('api', result=dict(r, completed_at=now-self.c['metric_ttl']-1))
        self.assertNotIn('pad_omni_anomaly_score{', generate_latest(self.registry).decode())
        self.m.update('api', result=r)
        self.m.fail('api', 'inference')
        self.assertFalse(self.m.ready())
        self.assertIn('pad_omni_inference_errors_total{group="api"} 1.0', generate_latest(self.registry).decode())

    def test_malformed_result_is_not_ready(self):
        now = time.time()
        base = dict(score=1., threshold=2., anomaly=0, evaluated_at=now,
                    completed_at=now, model_trained_at=now)
        for r in [{}, {'score': 1}, dict(base, score=float('nan')),
                  dict(base, threshold=float('inf')), dict(base, anomaly=1)]:
            self.m.update('api', result=r)
            self.assertFalse(self.m.ready())
            self.assertNotIn('pad_omni_anomaly_score{', generate_latest(self.registry).decode())

    def test_training_failure_does_not_remove_current_result(self):
        now = time.time()
        self.m.update('api', result=dict(score=1, threshold=3, anomaly=0,
            evaluated_at=now, completed_at=now, model_trained_at=now))
        self.m.fail('api', 'training')
        self.assertTrue(self.m.ready())


if __name__ == '__main__':
    unittest.main()
