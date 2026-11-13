"""TF1 adapter. Import only in the isolated, timeout-controlled worker."""
from types import SimpleNamespace
import numpy as np


class Backend:
    def __init__(self, c, dimensions):
        import tensorflow as tf
        from omni_anomaly.model import OmniAnomaly
        from omni_anomaly.training import Trainer
        from omni_anomaly.prediction import Predictor
        self.tf, self.c = tf, c
        self.graph = tf.Graph()
        cfg = SimpleNamespace(x_dim=dimensions, z_dim=c['z_dim'],
            window_length=c['window_length'], rnn_num_hidden=c['hidden'],
            dense_dim=c['dense'], nf_layers=c['nf_layers'],
            posterior_flow_type='nf', use_connected_z_q=True,
            use_connected_z_p=True, std_epsilon=1e-4, get_score_on_dim=False)
        with self.graph.as_default():
            np.random.seed(c['seed'])
            tf.set_random_seed(c['seed'])
            with tf.variable_scope('model') as scope:
                model = OmniAnomaly(config=cfg, name='model')
                # Identical graph construction order for training and restore.
                self.trainer = Trainer(model, model_vs=scope,
                    max_epoch=c['max_epoch'], batch_size=c['batch_size'],
                    valid_batch_size=c['batch_size'], valid_step_freq=10,
                    initial_lr=0.001, lr_anneal_epochs=40,
                    lr_anneal_factor=0.5, grad_clip_norm=10.)
                self.predictor = Predictor(model, batch_size=c['batch_size'],
                    n_z=c['n_z'], last_point_only=True)
                # Upstream builds the inference graph lazily; build before Saver.
                self.predictor._get_score_without_y()
            self.saver = tf.train.Saver(max_to_keep=1)
            self.init = tf.global_variables_initializer()
        self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(
            intra_op_parallelism_threads=c['tf_threads'],
            inter_op_parallelism_threads=c['tf_threads'],
            device_count={'GPU': 0}))
        self.session.run(self.init)

    def fit(self, values):
        with self.graph.as_default(), self.session.as_default():
            return self.trainer.fit(values, valid_portion=0.2)

    def scores(self, values):
        w, b = self.c['window_length'], self.c['batch_size']
        if len(values) < w:
            raise ValueError('not enough rows for a model window')
        result = []
        with self.graph.as_default(), self.session.as_default():
            for start in range(0, len(values)-w+1, b):
                score, _, _ = self.predictor.get_score(values[start:start+b+w-1])
                result.extend(-np.asarray(score, dtype=np.float64).reshape(-1))
        result = np.asarray(result)
        if len(result) != len(values)-w+1 or not np.isfinite(result).all():
            raise ValueError('invalid model scores')
        return result

    def save(self, prefix):
        with self.graph.as_default(), self.session.as_default():
            self.saver.save(self.session, str(prefix), write_meta_graph=False)

    def restore(self, prefix):
        with self.graph.as_default(), self.session.as_default():
            self.saver.restore(self.session, str(prefix))

    def close(self):
        self.session.close()
