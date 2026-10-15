"""VCM model architectures.

Candidates from the pre-registered experiment matrix (docs/decisions.md):

    A0  DS-CNN (32, 64, 128, 256)   ~57k params   baseline
    A1  DS-CNN (64, 128, 256, 256)  ~123k params  more capacity
    A2  TC-ResNet w32               ~131k params  residual temporal conv
    A5  old DS-CNN2D compact        ~50-60k params old 10-intent champion family

Input: (B, 1, 80, T) log-mel. Output contract:

    single head:  (B, n_way_head) intent logits
    multi-head:   {"intent": (B, n_way_head),
                   "slots":  {intent_name: (B, n_slot_i)}}

Slotted models carry one slot head per slotted intent. `forward` returns all
slot heads batched by intent; the trainer/evaluator masks rows by ground-truth
intent before computing slot loss/accuracy.

torch is imported lazily so this module is safe to import without the 'train'
extra (the architecture is only *instantiated* when training/exporting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ArchSpec:
    name: str                 # architecture family id
    channels: List[int]       # e.g. [32, 64, 128, 256]
    width: int = 32           # for tc_resnet
    n_way_head: int = 20

    @classmethod
    def a0(cls, n_way: int = 20) -> "ArchSpec":
        return cls("ds_cnn", [32, 64, 128, 256], width=32, n_way_head=n_way)

    @classmethod
    def a1(cls, n_way: int = 20) -> "ArchSpec":
        return cls("ds_cnn", [64, 128, 256, 256], width=32, n_way_head=n_way)

    @classmethod
    def a2(cls, n_way: int = 20) -> "ArchSpec":
        return cls("tc_resnet", [32, 32, 32, 32], width=32, n_way_head=n_way)

    @classmethod
    def a3(cls, n_way: int = 20) -> "ArchSpec":
        """CRNN: 1-D temporal conv stack -> BiGRU(128) -> intent head.

        `width`=conv width (32); `channels` repurposed to store the GRU hidden
        size (128) so the evaluator can reconstruct the spec from metadata.
        """
        return cls("crnn", [128], width=32, n_way_head=n_way)

    @classmethod
    def a4(cls, n_way: int = 20) -> "ArchSpec":
        """BC-ResNet small (Kim et al., Interspeech 2021, arXiv:2106.04140).

        Efficient edge KWS: 1-D temporal-conv stem + 2-D time-frequency
        BC (basic-cell) residual blocks, global-pool -> intent head. Targets
        OOS robustness at a small param/FLOP budget.

        `channels` stores the per-block channel list (4 BC blocks); `width`
        is the stem width (32). ~400-600K params.
        """
        return cls("bc_resnet", [32, 64, 128, 128], width=32, n_way_head=n_way)

    @classmethod
    def a5(cls, n_way: int = 20) -> "ArchSpec":
        """Compact depthwise-separable DS-CNN2D from the old 10-intent champions."""
        return cls("ds_cnn_sep", [32, 64, 128, 256], width=32, n_way_head=n_way)

    @classmethod
    def a6(cls, n_way: int = 20) -> "ArchSpec":
        """Small 2-D residual CNN: tests spatial residual hierarchy."""
        return cls("resnet2d_small", [24, 48, 96, 128], width=32, n_way_head=n_way)

    @classmethod
    def a7(cls, n_way: int = 20) -> "ArchSpec":
        """MobileNetV2-style inverted residual CNN: tests Pi-friendly efficiency."""
        return cls("mobilenetv2_tiny", [24, 32, 64, 96], width=32, n_way_head=n_way)

    @classmethod
    def a8(cls, n_way: int = 20) -> "ArchSpec":
        """Attention-pooled CRNN: tests temporal sequence modelling with learned pooling."""
        return cls("crnn_attn", [128], width=48, n_way_head=n_way)

    @classmethod
    def a9(cls, n_way: int = 20) -> "ArchSpec":
        """Tiny conformer/Transformer-style temporal encoder."""
        return cls("conformer_tiny", [128], width=64, n_way_head=n_way)

    @classmethod
    def a10(cls, n_way: int = 20) -> "ArchSpec":
        """Depthwise-separable 2-D ResNet with SE channel attention."""
        return cls("ds_resnet_se", [32, 64, 128, 160], width=32, n_way_head=n_way)

    @classmethod
    def a11(cls, n_way: int = 20) -> "ArchSpec":
        """MatchboxNet-lite: time-channel separable temporal residual CNN."""
        return cls("matchboxnet", [64, 64, 96, 128], width=64, n_way_head=n_way)

    @classmethod
    def a12(cls, n_way: int = 20) -> "ArchSpec":
        """MobileNetV3-Small adapted for log-mel input (Howard et al., 2019).

        Inverted residual blocks with squeeze-and-excitation (SE) channel
        attention and linear bottlenecks; h-swish/h-hardswish activations.
        Tests whether a compact mobile architecture approaches the a6/a10
        finalists at lower Pi cost. `channels` stores the per-stage output
        channel list (5 stages); `width` is the stem width (16). ~0.5-0.8M.
        """
        return cls("mobilenetv3_small", [16, 24, 48, 64, 96], width=16, n_way_head=n_way)


def _backbone(spec: ArchSpec, torch, nn):
    """Shared feature backbone -> (B, C, H, W) tensor."""
    if spec.name == "ds_cnn":
        ch = spec.channels
        layers, in_ch = [], 1
        for i, c in enumerate(ch):
            dil = 2 ** i
            layers += [
                nn.Conv2d(in_ch, c, kernel_size=3, dilation=dil, padding=dil),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.2),
            ]
            in_ch = c
        return nn.Sequential(*layers)

    if spec.name == "ds_cnn_sep":
        class DSBlock(nn.Module):
            def __init__(self, ch_in, ch_out, stride=1):
                super().__init__()
                self.depthwise = nn.Conv2d(
                    ch_in, ch_in, 3, stride=stride, padding=1,
                    groups=ch_in, bias=False,
                )
                self.pointwise = nn.Conv2d(ch_in, ch_out, 1, bias=False)
                self.bn = nn.BatchNorm2d(ch_out)
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.act(self.bn(self.pointwise(self.depthwise(x))))

        layers, in_ch = [], 1
        for i, c in enumerate(spec.channels):
            stride = 2 if i in (1, 3) else 1
            layers.append(DSBlock(in_ch, c, stride=stride))
            in_ch = c
        return nn.Sequential(*layers)

    if spec.name == "tc_resnet":
        w = spec.width

        class ResBlock(nn.Module):
            def __init__(self, ch, dil):
                super().__init__()
                self.c1 = nn.Conv2d(ch, ch, 3, padding=dil, dilation=dil)
                self.bn1 = nn.BatchNorm2d(ch)
                self.c2 = nn.Conv2d(ch, ch, 3, padding=dil, dilation=dil)
                self.bn2 = nn.BatchNorm2d(ch)
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                y = self.act(self.bn1(self.c1(x)))
                y = self.bn2(self.c2(y))
                return self.act(x + y)

        seq = [
            nn.Conv2d(1, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            *[ResBlock(w, 2 ** i) for i in range(4)],
        ]
        return nn.Sequential(*seq)

    if spec.name == "crnn":
        # 1-D temporal CRNN. Input (B,1,C,T) -> collapse channel dim, run a
        # dilated 1-D conv stack (temporal receptive field grows), then a
        # bidirectional GRU over time, read the final step.
        w = spec.width
        gru_hidden = spec.channels[0] if spec.channels else 128
        in_ch = 80  # feature channels (80 mel/MFCC bins); overridden in forward

        class CRNNBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(in_ch, w, 3, padding=1), nn.BatchNorm1d(w), nn.ReLU(inplace=True),
                )
                self.blocks = nn.Sequential(
                    nn.Conv1d(w, w, 3, dilation=2, padding=2), nn.BatchNorm1d(w), nn.ReLU(inplace=True),
                    nn.Conv1d(w, w, 3, dilation=4, padding=4), nn.BatchNorm1d(w), nn.ReLU(inplace=True),
                    nn.Conv1d(w, w, 3, dilation=8, padding=8), nn.BatchNorm1d(w), nn.ReLU(inplace=True),
                    nn.Conv1d(w, w, 3, dilation=16, padding=16), nn.BatchNorm1d(w), nn.ReLU(inplace=True),
                )
                self.gru = nn.GRU(w, gru_hidden, num_layers=1,
                                  bidirectional=True, batch_first=True)

            def forward(self, x):
                # x: (B, 1, C, T) -> (B, 1, T) -> (B, 1, T)
                xb = x[:, 0, :, :]                       # (B, C, T)
                h = self.stem(xb)
                h = self.blocks(h)
                h = h.permute(0, 2, 1)                   # (B, T, C) for GRU
                h = self.gru(h)[0]                       # (B, T, 2*gru_hidden)
                return h[:, -1, :]                        # (B, 2*gru_hidden)

        return CRNNBackbone()

    if spec.name == "bc_resnet":
        # BC-ResNet (Kim et al., Interspeech 2021, arXiv:2106.04140).
        # 1-D temporal-conv stem -> 4 2-D BC (basic-cell) residual blocks with
        # progressive channel growth + stride-2 downsampling -> global pool.
        # BC cell = two 3x3 convs (one dilated) + residual shortcut, the unit
        # the paper uses for efficient time-frequency modelling on edge KWS.
        stem_w = spec.width
        ch_list = spec.channels or [32, 64, 128, 128]

        class BCCell(nn.Module):
            """One basic cell: 3x3 conv (d=1) -> BN -> ReLU -> 3x3 conv (d=2)
            -> BN, with a 1x1 projection shortcut when channels/downsample."""
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.c1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.c2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=2, dilation=2)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.act = nn.ReLU(inplace=True)
                need_proj = (in_ch != out_ch) or (stride != 1)
                self.proj = (
                    nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                  nn.BatchNorm2d(out_ch))
                    if need_proj else nn.Identity()
                )

            def forward(self, x):
                y = self.act(self.bn1(self.c1(x)))
                y = self.bn2(self.c2(y))
                return self.act(y + self.proj(x))

        class BCResNetBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                # Temporal stem over the mel: 1x3 conv expanding the singleton
                # channel to stem_w, mirroring the paper's temporal-conv front
                # end. Output (B, stem_w, C, T) feeds the 2-D BC cells.
                self.stem = nn.Sequential(
                    nn.Conv2d(1, stem_w, kernel_size=(1, 3), padding=(0, 1)),
                    nn.BatchNorm2d(stem_w),
                    nn.ReLU(inplace=True),
                )
                cells, in_ch = [], stem_w
                for i, c in enumerate(ch_list):
                    stride = 2 if i in (1, 2) else 1  # downsample twice
                    cells.append(BCCell(in_ch, c, stride=stride))
                    in_ch = c
                self.cells = nn.Sequential(*cells)
                self.pool = nn.AdaptiveAvgPool2d(1)

            def forward(self, x):
                # x: (B, 1, C, T)
                h = self.stem(x)              # (B, stem_w, C, T)
                h = self.cells(h)             # (B, out_ch, H, W)
                h = self.pool(h).flatten(1)   # (B, out_ch)
                return h

        return BCResNetBackbone()

    if spec.name == "resnet2d_small":
        ch_list = spec.channels or [24, 48, 96, 128]

        class Res2DBlock(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1, drop=0.1):
                super().__init__()
                self.c1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.act = nn.ReLU(inplace=True)
                self.drop = nn.Dropout2d(drop)
                self.proj = (
                    nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                  nn.BatchNorm2d(out_ch))
                    if in_ch != out_ch or stride != 1 else nn.Identity()
                )

            def forward(self, x):
                y = self.act(self.bn1(self.c1(x)))
                y = self.drop(self.bn2(self.c2(y)))
                return self.act(y + self.proj(x))

        layers = [
            nn.Conv2d(1, ch_list[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_list[0]),
            nn.ReLU(inplace=True),
        ]
        in_ch = ch_list[0]
        for i, c in enumerate(ch_list):
            stride = 2 if i > 0 else 1
            layers.append(Res2DBlock(in_ch, c, stride=stride))
            layers.append(Res2DBlock(c, c, stride=1))
            in_ch = c
        return nn.Sequential(*layers)

    if spec.name == "mobilenetv2_tiny":
        ch_list = spec.channels or [24, 32, 64, 96]

        class InvertedResidual(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1, expand=4):
                super().__init__()
                mid = in_ch * expand
                self.use_res = stride == 1 and in_ch == out_ch
                self.block = nn.Sequential(
                    nn.Conv2d(in_ch, mid, 1, bias=False),
                    nn.BatchNorm2d(mid),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
                    nn.BatchNorm2d(mid),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(mid, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

            def forward(self, x):
                y = self.block(x)
                return x + y if self.use_res else y

        layers = [
            nn.Conv2d(1, spec.width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(spec.width),
            nn.ReLU6(inplace=True),
        ]
        in_ch = spec.width
        for i, c in enumerate(ch_list):
            stride = 2 if i in (1, 2, 3) else 1
            layers.append(InvertedResidual(in_ch, c, stride=stride, expand=4))
            layers.append(InvertedResidual(c, c, stride=1, expand=4))
            in_ch = c
        return nn.Sequential(*layers)

    if spec.name == "crnn_attn":
        w = spec.width
        gru_hidden = spec.channels[0] if spec.channels else 128

        class CRNNAttnBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(80, w, 5, padding=2, bias=False),
                    nn.BatchNorm1d(w),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(w, w, 5, dilation=2, padding=4, bias=False),
                    nn.BatchNorm1d(w),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(w, w, 3, dilation=4, padding=4, bias=False),
                    nn.BatchNorm1d(w),
                    nn.ReLU(inplace=True),
                )
                self.gru = nn.GRU(w, gru_hidden, num_layers=1,
                                  bidirectional=True, batch_first=True)
                self.attn = nn.Sequential(
                    nn.Linear(2 * gru_hidden, gru_hidden),
                    nn.Tanh(),
                    nn.Linear(gru_hidden, 1),
                )

            def forward(self, x):
                h = self.conv(x[:, 0, :, :]).transpose(1, 2)
                h = self.gru(h)[0]
                a = torch.softmax(self.attn(h), dim=1)
                return (h * a).sum(dim=1)

        return CRNNAttnBackbone()

    if spec.name == "conformer_tiny":
        d_model = spec.channels[0] if spec.channels else 128
        nhead = 4

        class ConformerTinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.subsample = nn.Sequential(
                    nn.Conv2d(1, spec.width, 3, stride=(2, 2), padding=1, bias=False),
                    nn.BatchNorm2d(spec.width),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(spec.width, spec.width, 3, stride=(2, 2), padding=1, bias=False),
                    nn.BatchNorm2d(spec.width),
                    nn.ReLU(inplace=True),
                )
                # Input windows are fixed at 80 mel bins. Two stride-2
                # frequency reductions leave 20 bins, so each time step has
                # spec.width * 20 channels before projection.
                self.proj = nn.Linear(spec.width * 20, d_model)
                enc = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 2,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc, num_layers=2)
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x):
                h = self.subsample(x)             # (B, C, F, T)
                h = h.permute(0, 3, 1, 2).flatten(2)
                h = self.proj(h)
                h = self.encoder(h)
                return self.norm(h.mean(dim=1))

        return ConformerTinyBackbone()

    if spec.name == "ds_resnet_se":
        ch_list = spec.channels or [32, 64, 128, 160]

        class SE(nn.Module):
            def __init__(self, ch, reduction=8):
                super().__init__()
                mid = max(ch // reduction, 4)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Conv2d(ch, mid, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid, ch, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return x * self.fc(self.pool(x))

        class DSSEBlock(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1, drop=0.08):
                super().__init__()
                self.dw1 = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                                     groups=in_ch, bias=False)
                self.pw1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.dw2 = nn.Conv2d(out_ch, out_ch, 3, padding=1,
                                     groups=out_ch, bias=False)
                self.pw2 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.se = SE(out_ch)
                self.drop = nn.Dropout2d(drop)
                self.act = nn.ReLU(inplace=True)
                self.proj = (
                    nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                  nn.BatchNorm2d(out_ch))
                    if in_ch != out_ch or stride != 1 else nn.Identity()
                )

            def forward(self, x):
                y = self.act(self.bn1(self.pw1(self.dw1(x))))
                y = self.bn2(self.pw2(self.dw2(y)))
                y = self.drop(self.se(y))
                return self.act(y + self.proj(x))

        layers = [
            nn.Conv2d(1, ch_list[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_list[0]),
            nn.ReLU(inplace=True),
        ]
        in_ch = ch_list[0]
        for i, c in enumerate(ch_list):
            stride = 2 if i in (1, 2, 3) else 1
            layers.append(DSSEBlock(in_ch, c, stride=stride))
            layers.append(DSSEBlock(c, c, stride=1))
            in_ch = c
        return nn.Sequential(*layers)

    if spec.name == "matchboxnet":
        ch_list = spec.channels or [64, 64, 96, 128]
        width = spec.width

        class MatchboxBlock(nn.Module):
            def __init__(self, in_ch, out_ch, kernel=11, dilation=1, repeats=2, drop=0.12):
                super().__init__()
                layers = []
                cur = in_ch
                pad = dilation * (kernel // 2)
                for _ in range(repeats):
                    layers += [
                        nn.Conv1d(cur, cur, kernel, padding=pad, dilation=dilation,
                                  groups=cur, bias=False),
                        nn.Conv1d(cur, out_ch, 1, bias=False),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(inplace=True),
                        nn.Dropout(drop),
                    ]
                    cur = out_ch
                self.block = nn.Sequential(*layers)
                self.proj = (
                    nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                                  nn.BatchNorm1d(out_ch))
                    if in_ch != out_ch else nn.Identity()
                )
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.act(self.block(x) + self.proj(x))

        class MatchboxBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(80, width, 1, bias=False),
                    nn.BatchNorm1d(width),
                    nn.ReLU(inplace=True),
                )
                blocks = []
                in_ch = width
                for c, dil in zip(ch_list, [1, 2, 4, 1]):
                    blocks.append(MatchboxBlock(in_ch, c, kernel=11, dilation=dil, repeats=2))
                    in_ch = c
                self.blocks = nn.Sequential(*blocks)

            def forward(self, x):
                h = self.stem(x[:, 0, :, :])
                h = self.blocks(h)
                return h.mean(dim=-1)

        return MatchboxBackbone()

    if spec.name == "mobilenetv3_small":
        # MobileNetV3-Small adapted to 2-D log-mel (B,1,80,T). Inverted
        # residual blocks with SE channel attention + linear bottlenecks and
        # h-swish/h-hardswish activations. Strides at stages 1,2,3 give a
        # 1/8 spatial/temporal reduction; the global pool in embed() handles
        # the remaining size.
        ch_list = spec.channels or [16, 24, 48, 64, 96]
        stem_w = spec.width or 16

        class HSwish(nn.Module):
            def forward(self, x):
                return x * torch.nn.functional.hardtanh(x + 3.0, inplace=False) / 6.0

        class HHardSwish(nn.Module):
            def forward(self, x):
                return x * torch.nn.functional.relu6(x + 3.0, inplace=False) / 6.0

        class SE3(nn.Module):
            def __init__(self, ch, reduction=4):
                super().__init__()
                mid = max(ch // reduction, 8)
                self.fc = nn.Sequential(
                    nn.Linear(ch, mid), HHardSwish(),
                    nn.Linear(mid, ch), HSwish(),
                )

            def forward(self, x):
                s = x.mean(dim=[2, 3])                          # (B, C)
                w = self.fc(s).unsqueeze(-1).unsqueeze(-1)      # (B, C, 1, 1)
                return x * w

        class MBv3IR(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1, expand=6, use_se=False, drop=0.1):
                super().__init__()
                mid = in_ch * expand
                self.use_res = stride == 1 and in_ch == out_ch
                self.exp = nn.Sequential(
                    nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), HSwish(),
                )
                self.dw = nn.Sequential(
                    nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
                    nn.BatchNorm2d(mid), HSwish(),
                )
                self.se = SE3(mid) if use_se else nn.Identity()
                self.proj = nn.Sequential(
                    nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
                )
                self.drop = nn.Dropout(drop)

            def forward(self, x):
                y = self.proj(self.se(self.dw(self.exp(x))))
                y = self.drop(y)
                return x + y if self.use_res else y

        layers = [
            nn.Conv2d(1, stem_w, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_w), HSwish(),
        ]
        in_ch = stem_w
        for i, c in enumerate(ch_list):
            stride = 2 if i in (1, 2, 3) else 1
            use_se = (i in (2, 4))
            layers.append(MBv3IR(in_ch, c, stride=stride, expand=6, use_se=use_se))
            in_ch = c
        return nn.Sequential(*layers)

    raise ValueError(f"Unknown arch: {spec.name}")


def build_model(spec: ArchSpec, slot_heads: Optional[dict] = None):
    """Instantiate a torch model for the given spec.

    `slot_heads` maps intent NAME -> list of slot class names (effective set,
    REJECT already stripped per slot_reject). When provided, the model exposes
    one slot head per slotted intent in addition to the intent head.
    Requires the 'train' extra (torch).
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Building a model requires torch. Install with: pip install 'hey-swift[train]'"
        ) from exc

    feat = _backbone(spec, torch, nn)
    if spec.name in {"ds_cnn", "ds_cnn_sep"}:
        out_ch = spec.channels[-1]
    elif spec.name == "crnn":
        out_ch = 2 * (spec.channels[0] if spec.channels else 128)  # BiGRU both dirs
    elif spec.name == "bc_resnet":
        out_ch = spec.channels[-1] if spec.channels else 128
    elif spec.name in {"resnet2d_small", "mobilenetv2_tiny", "ds_resnet_se", "matchboxnet", "mobilenetv3_small"}:
        out_ch = spec.channels[-1]
    elif spec.name in {"crnn_attn", "conformer_tiny"}:
        out_ch = 2 * (spec.channels[0] if spec.channels else 128) if spec.name == "crnn_attn" else (spec.channels[0] if spec.channels else 128)
    else:
        out_ch = spec.width

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = feat
            self.intent_head = nn.Linear(out_ch, spec.n_way_head)
            # Use nn.ModuleDict so slot heads are registered sub-modules and
            # are moved to the correct device by model.to(device).
            self.slot_head_map = nn.ModuleDict()
            if slot_heads:
                for intent_name, classes in slot_heads.items():
                    self.slot_head_map[intent_name] = nn.Linear(out_ch, len(classes))

        def embed(self, x):
            z = self.features(x)
            if z.dim() == 4:  # 2-D backbones: global pool over spatial dims
                z = z.mean(dim=[2, 3])
            return z

        def forward(self, x):
            z = self.embed(x)
            intent_logits = self.intent_head(z)
            if not self.slot_head_map:
                return intent_logits
            slot_logits = {name: head(z) for name, head in self.slot_head_map.items()}
            return {"intent": intent_logits, "slots": slot_logits}

    model = Model()
    # stash intent names for the per-row slot dispatch
    from .taxonomy import INTENTS
    model._names = list(INTENTS)
    return model
