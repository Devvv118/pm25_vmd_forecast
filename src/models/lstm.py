"""LSTM forecaster (PyTorch) implementing the shared Forecaster interface."""
import torch
import torch.nn as nn

from .base import Forecaster


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, n_lags, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(Forecaster):
    name = "LSTM"

    def __init__(self, hidden_size=32, num_layers=1, epochs=30, lr=1e-3,
                 batch_size=64, device=None):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        # Auto-detect: use a GPU if one is available, otherwise fall back
        # to CPU. Pass device="cpu" explicitly to force CPU regardless.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._mean = 0.0
        self._std = 1.0

    def fit(self, X_train, y_train):
        # z-score normalize using train statistics only, then unscale at
        # predict time. Without this the LSTM converges very poorly.
        self._mean = float(X_train.mean())
        self._std = float(X_train.std()) or 1.0

        X_s = (X_train - self._mean) / self._std
        y_s = (y_train - self._mean) / self._std

        self.model = _LSTMNet(self.hidden_size, self.num_layers).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X_s, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_t = torch.tensor(y_s, dtype=torch.float32).to(self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size, shuffle=True,
        )

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        X_s = (X - self._mean) / self._std
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_s, dtype=torch.float32).unsqueeze(-1).to(self.device)
            pred_s = self.model(X_t).cpu().numpy()
        return pred_s * self._std + self._mean
