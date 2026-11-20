"""IMU encoders: (B, L, 6) -> (B, L / stride, C). Register new ones in ENCODERS."""

import torch.nn as nn


class CNNGRUEncoder(nn.Module):
    """2 conv layers, then 2 GRU layers.

    The second conv downsamples by `stride`; the GRU is unidirectional, so a
    feature only sees the past of the window (causal at inference).
    """

    def __init__(self, in_ch=6, conv_ch=(32, 64), kernel=5, stride=5, hidden=128, dropout=0.0):
        super().__init__()
        c1, c2 = conv_ch
        self.stride = stride
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel, stride=1, padding=kernel // 2), nn.GELU(),
            nn.Conv1d(c1, c2, stride, stride=stride), nn.GELU(),
        )
        self.gru = nn.GRU(c2, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.out_dim = hidden

    def forward(self, x):
        f = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        return self.gru(f)[0]


ENCODERS = {'cnn_gru': CNNGRUEncoder}
