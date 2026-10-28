# SimpleMLP.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleMLPClassifier:
    def __init__(
        self,
        hidden=(128, 64),
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        epochs=30,
        device=None,
        seed=42,
        verbose=True,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_ = None
        self.input_dim_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.input_dim_ = int(X.shape[1])
        self.n_classes_ = int(np.max(y)) + 1

        self.model_ = SimpleMLP(
            input_dim=self.input_dim_,
            n_classes=self.n_classes_,
            hidden=self.hidden,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model_.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * xb.size(0)

            if self.verbose and (epoch == 1 or epoch % 5 == 0 or epoch == self.epochs):
                avg_loss = total_loss / len(ds)
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)

        self.model_.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self.device)
            logits = self.model_(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds
