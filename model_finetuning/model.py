# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from EGAT import EGAT  # EGAT 모듈 필요

class GraphBepiLightning(pl.LightningModule):
    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_u1 = nn.Linear(hidden_dim, hidden_dim)
        self.gat = EGAT(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.loss_fn = nn.BCELoss()

    def forward(self, x, edge_attr):
        x = F.relu(self.W_v(x))
        x, _ = self.gat(x, edge_attr)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        out = torch.sigmoid(self.mlp(x))
        return out

    def training_step(self, batch, batch_idx):
        x, edge_attr, y = batch
        y_hat = self(x, edge_attr)
        y_hat = y_hat.squeeze(-1) if y_hat.dim() > y.dim() else y_hat
        loss = self.loss_fn(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_attr, y = batch
        y_hat = self(x, edge_attr)
        y_hat = y_hat.squeeze(-1) if y_hat.dim() > y.dim() else y_hat
        loss = self.loss_fn(y_hat, y.float())
        self.log('val_loss', loss)
        return {'val_loss': loss, 'y_hat': y_hat, 'y': y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
