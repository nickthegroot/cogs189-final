import math
import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(pl.LightningModule):

    def __init__(
        self,
        hz: int,
        dim: int,
        nhead = 2,
        d_hid = 100,
        nlayers = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.d_model = int(.5 * hz) # 500ms chunks
        self.dim = dim


        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        self.encoder = nn.Linear(dim, self.d_model)
        self.decoder = nn.Linear(int(hz * 2.5 * dim * 2.5), 1)

        self.criterion = nn.BCEWithLogitsLoss()
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        seq_size = int(src.size(2) / self.dim)
        src = src.reshape((src.size(0), seq_size, self.dim))
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        src_mask = self.__generate_square_subsequent_mask(src.size(1))
        output = self.transformer_encoder(src, src_mask)
        output = output.flatten(1, -1)
        output = self.decoder(output)
        return output

    def __generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        This forces the filters to be CAUSAL: only looking at times in the past.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

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
        
        pred = torch.sigmoid(score).round()
        acc = (pred == y).float().mean()
        self.log('val_acc', acc)
        
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-5)