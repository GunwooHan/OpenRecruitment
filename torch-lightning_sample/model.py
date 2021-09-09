import os

import timm
import torch
from torch.nn import functional as F
import pytorch_lightning as pl


class CustomModel(pl.LightningModule):
    def __init__(self, model_name='tf_efficientnet_b0', num_classes=18):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=18)
        self.optimizer = self.configure_optimizers()

    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)      # Loss 수정
        self.log('val_loss', loss)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = FocalLoss()(y_hat, y)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self(x)
        loss = FocalLoss()(y_hat, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

