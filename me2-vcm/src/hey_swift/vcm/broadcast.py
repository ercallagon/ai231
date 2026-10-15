"""BC-ResNet-8 adaptation for 80-bin, two-second intent/slot classification.

Architecture reference: Kim et al., Interspeech 2021, arXiv:2106.04140.
See THIRD_PARTY_BCRESNET_LICENSE.txt for the reference implementation notice.
Unlike historical A4, these blocks actually broadcast temporal residuals.
"""

NAME = "bc_resnet_broadcast8"


def build_broadcast_model(n_way=20, slot_heads=None):
    import torch
    from torch import nn

    class SubSpectralNorm(nn.Module):
        def __init__(self, channels, groups=5):
            super().__init__()
            self.groups = groups
            self.norm = nn.BatchNorm2d(channels * groups)

        def forward(self, x):
            batch, channels, frequency, time = x.shape
            grouped = x.reshape(batch, channels * self.groups, frequency // self.groups, time)
            return self.norm(grouped).reshape(batch, channels, frequency, time)

    class BroadcastBlock(nn.Module):
        def __init__(self, input_channels, channels, stage, frequency_stride):
            super().__init__()
            self.identity = input_channels == channels and frequency_stride == 1
            self.transition = (nn.Identity() if input_channels == channels else nn.Sequential(
                nn.Conv2d(input_channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(),
            ))
            self.frequency = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 1), stride=(frequency_stride, 1),
                          padding=(1, 0), groups=channels, bias=False),
                SubSpectralNorm(channels),
            )
            dilation = 2 ** stage
            self.temporal = nn.Sequential(
                nn.Conv2d(channels, channels, (1, 3), padding=(0, dilation),
                          dilation=(1, dilation), groups=channels, bias=False),
                nn.BatchNorm2d(channels), nn.SiLU(),
                nn.Conv2d(channels, channels, 1, bias=False), nn.Dropout2d(0.1),
            )

        def forward(self, x):
            spatial = self.frequency(self.transition(x))
            temporal = self.temporal(spatial.mean(dim=2, keepdim=True))
            output = spatial + temporal
            if self.identity:
                output = output + x
            return torch.relu(output)

    class BroadcastModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 128, 5, stride=(2, 1), padding=2, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(),
            )
            blocks = []
            input_channels = 128
            for stage, (channels, repeats) in enumerate(zip((64, 96, 128, 160), (2, 2, 4, 4))):
                for repeat in range(repeats):
                    stride = 2 if stage in (1, 2) and repeat == 0 else 1
                    blocks.append(BroadcastBlock(input_channels, channels, stage, stride))
                    input_channels = channels
            self.blocks = nn.Sequential(*blocks)
            self.tail = nn.Sequential(
                nn.Conv2d(160, 160, 5, padding=(0, 2), groups=160, bias=False),
                nn.Conv2d(160, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            )
            self.intent_head = nn.Linear(256, n_way)
            self.slot_head_map = nn.ModuleDict({name: nn.Linear(256, len(values))
                                               for name, values in (slot_heads or {}).items()})

        def embed(self, x):
            return self.tail(self.blocks(self.stem(x))).mean(dim=(2, 3))

        def forward(self, x):
            embedding = self.embed(x)
            intent = self.intent_head(embedding)
            if not self.slot_head_map:
                return intent
            return {"intent": intent, "slots": {name: head(embedding) for name, head in self.slot_head_map.items()}}

    return BroadcastModel()
