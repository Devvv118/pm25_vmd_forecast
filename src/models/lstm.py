"""LSTM forecaster (PyTorch, CPU only) with seeding, validation-based early stopping
and optional exogenous inputs.

Input row layout: the first `n_seq` columns are the lag sequence fed to the
LSTM; any further columns (e.g. station count) are concatenated to the last
hidden state before the output layer.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from .base import Forecaster


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size, num_layers, n_extra):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size + n_extra, 1)

    def forward(self, seq, extra):
        out, _ = self.lstm(seq)
        h = out[:, -1, :]
        if extra.shape[1]:
            h = torch.cat([h, extra], dim=1)
        return self.fc(h).squeeze(-1)


class LSTMForecaster(Forecaster):
    name = "LSTM"

    def __init__(self, n_seq=24, hidden_size=32, num_layers=1,
                 epochs=100, patience=10, lr=1e-3, batch_size=64, seed=42):
        self.n_seq = n_seq
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.epochs, self.patience, self.lr = epochs, patience, lr
        self.batch_size, self.seed = batch_size, seed
        self.model = None
        self._mean, self._std = 0.0, 1.0
        self._ex_mean = self._ex_std = None
        self.best_epoch_ = None

    def _prep(self, X):
        seq = (X[:, :self.n_seq] - self._mean) / self._std
        extra = X[:, self.n_seq:]
        if extra.shape[1]:
            extra = (extra - self._ex_mean) / self._ex_std
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)
        ex_t = torch.tensor(extra, dtype=torch.float32)
        return seq_t, ex_t

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # z-score with TRAIN statistics only
        self._mean = float(X_train[:, :self.n_seq].mean())
        self._std = float(X_train[:, :self.n_seq].std()) or 1.0
        if X_train.shape[1] > self.n_seq:
            ex = X_train[:, self.n_seq:]
            self._ex_mean, self._ex_std = ex.mean(0), np.where(ex.std(0) == 0, 1.0, ex.std(0))

        seq, extra = self._prep(X_train)
        y_t = torch.tensor((y_train - self._mean) / self._std, dtype=torch.float32)
        n_extra = extra.shape[1]
        self.model = _LSTMNet(self.hidden_size, self.num_layers, n_extra)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(seq, extra, y_t),
            batch_size=self.batch_size, shuffle=True, generator=gen)

        if X_val is not None:
            v_seq, v_ex = self._prep(X_val)
            v_y = torch.tensor((y_val - self._mean) / self._std, dtype=torch.float32)

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for sb, eb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model(sb, eb), yb).backward()
                opt.step()
            if X_val is None:
                continue
            self.model.eval()
            with torch.no_grad():
                val = float(loss_fn(self.model(v_seq, v_ex), v_y))
            if val < best - 1e-9:
                best, bad, self.best_epoch_ = val, 0, epoch
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.val_mse_ = best * self._std ** 2 if X_val is not None else None
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            seq, extra = self._prep(X)
            pred_s = self.model(seq, extra).numpy()
        return pred_s * self._std + self._mean
