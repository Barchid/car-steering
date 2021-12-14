from argparse import ArgumentParser
from os import times

import torch
import pytorch_lightning as pl
from torch.nn import functional as F

from project.models.pilotnet import PilotNet


class DrivingModule(pl.LightningModule):
    def __init__(self, learning_rate: float, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = PilotNet(in_channels=3, out_channels=1)

    def forward(self, x):

        # (RGB frame) -> (predicted angle)
        angle = self.model(x)

        return angle

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log('train_mse', loss, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        # logs
        self.log('val_mse', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log('test_mse', loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # TODO: change the optimizer if you want
        # exemple: Stochastic Gradient Descent (SGD) with weight decay, etc
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # Here, you add every arguments needed for your module
        # NOTE: they must appear as arguments in the __init___() function
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser
