import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from trans_model import PositionalEncoding
from torch import nn, optim


class SeparableConv2d(nn.Module):
    """
    Depthwise Conv folloowed by Pointwise Conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int],
        **args,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, groups=in_channels, padding=padding
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, X: torch.Tensor):
        X = self.depthwise(X)
        X = self.pointwise(X)
        return X


class EEGNet(nn.Module):
    """
    EEGNet PyTorch implentation.

    Based off https://arxiv.org/abs/1611.08024
    """

    def __init__(
        self,
        C: int,  # Number of (Input) Channels
        T: int,  # Number of Time points
        F1: int,  # Number of Temporal Filters
        D: int,  # Number of Spatial Filters
        F2: int,  # Number of Pointwise Filters
    ) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 25), padding=(0, 12)),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1),  # DepthwiseConv2D
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )

        self.block2 = nn.Sequential(
            SeparableConv2d(D * F1, F2, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
            nn.Flatten(),
        )

        self.output_size = F2 * (T // 32)

    def forward(self, X: torch.Tensor, **args):
        """
        Expected Input: (N x C x T)
        Expected Output: (N x (F2 * T // 32))
        """
        X = X.unsqueeze(-3)
        X = self.block1(X)
        X = self.block2(X)
        return X


class EEGTransformer(pl.LightningModule):
    def __init__(
        self,
        C: int,  # Number of (Input) Channels
        T: int,  # Number of Time points
        F1=8,  # Number of Temporal Filters
        D=2,  # Number of Spatial Filters
        F2=16,  # Number of Pointwise Filters
        heads=2, # Number of Attentional Heads
    ) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(T)
        self.encoder = EEGNet(C, T, F1, D, F2)
        self.attn = nn.MultiheadAttention(
            self.encoder.output_size,
            heads,
            batch_first=True,
        )
        self.fc = nn.Linear(self.encoder.output_size, 1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, X: torch.Tensor):
        X = self.pos_encoder(X)
        X = self.encoder(X)
        X = self.attn(X, X, X)[0] # self-attention
        X = self.fc(X)
        return X

    def training_step(self, batch, batch_idx):
        X, y = batch

        score = self.forward(X)
        loss = self.criterion(score, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        score = self.forward(X)
        loss = self.criterion(score, y)
        self.log('val_loss', loss)
        
        pred = F.sigmoid(score).round()
        acc = (pred == y).float().mean()
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
