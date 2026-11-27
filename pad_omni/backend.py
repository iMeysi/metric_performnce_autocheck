"""Experimental recurrent VAE + planar flows, NOT a bit-equivalent upstream port.
Imported only by worker. CPU-only; checkpoints are incompatible with TF1.
"""
import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BACKEND_ID = 'torch-rvae-planar-v1'


def normal_log_prob(x, mean, log_std):
    return (-0.5 * ((x - mean) * torch.exp(-log_std)).square()
            - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)


class PlanarFlow(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.u = nn.Parameter(torch.randn(size) * 0.01)
        self.w = nn.Parameter(torch.randn(size) * 0.01)
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, z):
        wu = (self.w * self.u).sum()
        # Invertibility constraint w^T u_hat > -1.
        u = self.u + ((-1 + F.softplus(wu)) - wu) * self.w / (self.w.square().sum() + 1e-8)
        h = torch.tanh(z @ self.w + self.b)
        out = z + h.unsqueeze(-1) * u
        det = 1 + (1 - h.square()) * (self.w * u).sum()
        return out, torch.log(det.abs().clamp_min(1e-8))


class RecurrentVAE(nn.Module):
    def __init__(self, c, dimensions):
        super().__init__()
        h, d, z = c['hidden'], c['dense'], c['z_dim']
        self.z_dim, self.hidden = z, h
        self.encoder = nn.GRU(dimensions, h, batch_first=True)
        self.posterior = nn.GRUCell(h + z, h)
        self.q_head = nn.Sequential(nn.Linear(h, d), nn.Tanh(), nn.Linear(d, 2*z))
        self.prior = nn.Sequential(nn.Linear(z, d), nn.Tanh(), nn.Linear(d, 2*z))
        self.flows = nn.ModuleList([PlanarFlow(z) for _ in range(c['nf_layers'])])
        self.decoder = nn.GRU(z, h, batch_first=True)
        self.x_head = nn.Sequential(nn.Linear(h, d), nn.Tanh(), nn.Linear(d, 2*dimensions))

    @staticmethod
    def parameters_normal(head):
        mean, log_std = head.chunk(2, dim=-1)
        return mean, log_std.clamp(-7, 5)

    def forward(self, x, noise=None):
        enc, _ = self.encoder(x)
        previous = x.new_zeros(x.shape[0], self.z_dim)
        state = x.new_zeros(x.shape[0], self.hidden)
        zs, log_qs, log_ps = [], [], []
        for t in range(x.shape[1]):
            state = self.posterior(torch.cat((enc[:, t], previous), -1), state)
            mu, ls = self.parameters_normal(self.q_head(state))
            eps = torch.randn_like(mu) if noise is None else noise[t].expand_as(mu)
            z = mu + torch.exp(ls) * eps
            log_q = normal_log_prob(z, mu, ls)
            for flow in self.flows:
                z, log_det = flow(z)
                log_q = log_q - log_det
            if t == 0:
                pm, pl = torch.zeros_like(z), torch.zeros_like(z)
            else:
                pm, pl = self.parameters_normal(self.prior(previous))
            log_ps.append(normal_log_prob(z, pm, pl))
            log_qs.append(log_q)
            zs.append(z)
            previous = z
        decoded, _ = self.decoder(torch.stack(zs, 1))
        xm, xl = self.parameters_normal(self.x_head(decoded))
        reconstruction = normal_log_prob(x, xm, xl)
        loss = (torch.stack(log_qs, 1) - torch.stack(log_ps, 1) - reconstruction).mean()
        return loss, reconstruction


class Backend:
    def __init__(self, c, dimensions):
        self.c, self.dimensions = c, dimensions
        torch.set_num_threads(c['torch_threads'])
        torch.manual_seed(c['seed'])
        self.model = RecurrentVAE(c, dimensions).cpu()
        self.metadata = {'backend': BACKEND_ID, 'dimensions': dimensions,
                         'architecture': {k: c[k] for k in
                             ('hidden', 'dense', 'z_dim', 'nf_layers', 'window_length')}}

    def batches(self, values, shuffle=False):
        values = np.asarray(values, dtype=np.float32)
        w, b = self.c['window_length'], self.c['batch_size']
        if values.ndim != 2 or values.shape[1] != self.dimensions or not np.isfinite(values).all():
            raise ValueError('invalid model input')
        if len(values) < w:
            raise ValueError('not enough rows for a model window')
        indices = np.arange(len(values) - w + 1)
        if shuffle:
            np.random.default_rng(self.c['seed'] + self.epoch).shuffle(indices)
        for start in range(0, len(indices), b):
            yield torch.from_numpy(np.stack([values[i:i+w] for i in indices[start:start+b]]))

    def fit(self, values):
        cut = len(values) - int(len(values) * 0.2)
        training, validation = values[:cut], values[cut:]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        best, best_state = float('inf'), None
        generator = torch.Generator().manual_seed(self.c['seed'] + 1)
        noise = torch.randn(self.c['window_length'], self.c['z_dim'], generator=generator)
        for self.epoch in range(self.c['max_epoch']):
            self.model.train()
            for batch in self.batches(training, shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = self.model(batch)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
            self.model.eval()
            total, count = 0., 0
            with torch.no_grad():
                for batch in self.batches(validation):
                    loss, _ = self.model(batch, noise)
                    total += float(loss) * len(batch)
                    count += len(batch)
            metric = total / count
            if not math.isfinite(metric):
                raise ValueError('nonfinite validation loss')
            if metric < best:
                best, best_state = metric, copy.deepcopy(self.model.state_dict())
        if best_state is None:
            raise ValueError('no valid trained model')
        self.model.load_state_dict(best_state)
        return {'validation_negative_elbo': best}

    def scores(self, values):
        self.model.eval()
        # Fixed per-window draws: calibration and online score do not depend on
        # batch size, window position, or unrelated requests. MC mean log p(x|z).
        rng = torch.Generator().manual_seed(self.c['seed'] + 2)
        noises = torch.randn(self.c['n_z'], self.c['window_length'], self.c['z_dim'], generator=rng)
        result = []
        with torch.no_grad():
            for batch in self.batches(values):
                scores = torch.stack([self.model(batch, noise)[1][:, -1] for noise in noises])
                result.extend((-scores.mean(0)).double().tolist())
        result = np.asarray(result, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError('invalid model scores')
        return result

    def save(self, prefix):
        torch.save({'metadata': self.metadata, 'state': self.model.state_dict()}, str(prefix) + '.pt')

    def restore(self, prefix):
        checkpoint = torch.load(str(prefix) + '.pt', map_location='cpu', weights_only=True)
        if checkpoint['metadata'] != self.metadata:
            raise ValueError('incompatible checkpoint; retrain required')
        self.model.load_state_dict(checkpoint['state'], strict=True)
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError('nonfinite checkpoint parameters')
        self.model.eval()

    def close(self):
        self.model = None
