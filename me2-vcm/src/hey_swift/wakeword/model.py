"""Wake-word model architecture: WakeGRU (faithful exp15 port).

Faithful port of `me2-wakeword-swift/scripts/model_swift.py`. The historical
exp15 champion is a 2-class WakeGRU (3x ConvBlock -> GRU over time -> FC head),
204,738 params, input (B, 80, 100) [40-mel x (static+delta)], output (B, 2)
logits where index 0 = wake, 1 = not-wake.

Trained from scratch (no pretrained weights). torch is imported lazily so the
module imports cleanly on the Pi (inference uses the exported ONNX, not torch).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WakeArchSpec:
    """Frozen architectural spec for the exp15 WakeGRU (2-class)."""
    n_mels: int = 40
    frames: int = 100
    channels: tuple = (16, 32, 64)
    gru_hidden: int = 128
    dropout: float = 0.4
    fc_hidden: int = 64
    n_classes: int = 2
    in_channels: int = 2
    target_params: int = 204738


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Building the wake-word model requires torch: "
            "pip install 'hey-swift[train]'"
        ) from exc
    import torch
    import torch.nn as nn
    return torch, nn


def build_model(spec: WakeArchSpec | None = None):
    """Instantiate the exp15 WakeGRU (requires torch). Returns an nn.Module."""
    spec = spec or WakeArchSpec()
    torch, nn = _require_torch()

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout, k, s, p):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
            self.bn = nn.BatchNorm2d(cout)
            self.act = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(2, 2)

        def forward(self, x):
            return self.pool(self.act(self.bn(self.conv(x))))

    class WakeGRU(nn.Module):
        """3x ConvBlock -> GRU over time -> FC head.

        After 3 max-pools: mel axis = n_mels//8 (5 for 40), time = frames//8 (13).
        Flatten (c2, mel//8) per frame -> GRU input. Last timestep -> head.
        """

        def __init__(self, n_mels=40, frames=100, channels=(16, 32, 64),
                     gru_hidden=128, dropout=0.4, fc_hidden=64, n_classes=2,
                     in_channels=2):
            super().__init__()
            c0, c1, c2 = channels
            self.in_channels = in_channels
            self.b1 = ConvBlock(in_channels, c0, 3, 1, 1)
            self.b2 = ConvBlock(c0, c1, 3, 1, 1)
            self.b3 = ConvBlock(c1, c2, 3, 1, 1)
            self._mel_out = n_mels // 8
            self.gru = nn.GRU(input_size=c2 * self._mel_out, hidden_size=gru_hidden,
                              num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(gru_hidden, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden, n_classes),
            )
            self._c2 = c2
            self._frames_in = frames

        def forward(self, x):
            # x: (B, 2*N_MELS, F) -> (B, in_channels, N_MELS, F)
            if x.dim() == 3:
                B, C, F = x.shape
                parts = torch.chunk(x, self.in_channels, dim=1)
                x = torch.stack(parts, dim=1)
            x = self.b1(x)
            x = self.b2(x)
            x = self.b3(x)
            # (B, c2, H, T) -> (B, T, c2*H)
            x = x.permute(0, 3, 1, 2)
            x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3))
            out, _ = self.gru(x)
            x = out[:, -1, :]
            return self.head(x)

    return WakeGRU(
        n_mels=spec.n_mels, frames=spec.frames, channels=spec.channels,
        gru_hidden=spec.gru_hidden, dropout=spec.dropout, fc_hidden=spec.fc_hidden,
        n_classes=spec.n_classes, in_channels=spec.in_channels,
    )


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
