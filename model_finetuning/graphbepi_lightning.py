# graphbepi_lightning.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from model import GraphBepi  # GraphBepi 모델
from torch.optim import Adam

class GraphBepiLightning(pl.LightningModule):
    def __init__(self, hidden_dim=128, lr=1e-3):
        super(GraphBepiLightning, self).__init__()
        self.model = GraphBepi(hidden_dim=hidden_dim)
        self.loss_fn = nn.BCELoss()
        self.lr = lr

        # 출력 로그
        self.train_losses = []
        self.val_losses = []

    def forward(self, x, edge_attr):
        return self.model(x, edge_attr)

    def training_step(self, batch, batch_idx):
        feat, edge, label = batch
        pred, _ = self(feat, edge)
        loss = self.loss_fn(pred, label)
        self.log('train_loss', loss, prog_bar=True)
        self.train_losses.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        feat, edge, label = batch
        pred, _ = self(feat, edge)
        loss = self.loss_fn(pred, label)
        self.log('val_loss', loss, prog_bar=True)
        self.val_losses.append(loss.detach())
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    # v2.0 이후 validation_epoch_end 대체
    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_losses).mean()
        self.log('avg_val_loss', avg_val_loss)
        self.val_losses.clear()
