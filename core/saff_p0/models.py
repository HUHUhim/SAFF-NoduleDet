from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import SUPPORTED_SAFF_VARIANTS


def conv_norm_act(in_ch: int, out_ch: int, k: int = 3, s: int = 1, groups: int = 1) -> nn.Sequential:
    p = k // 2
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.SiLU(inplace=True),
    )


class SE3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv3d(channels, hidden, 1)
        self.fc2 = nn.Conv3d(hidden, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool3d(x, 1)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class MBConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 2, kernel: int = 3):
        super().__init__()
        mid = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            conv_norm_act(in_ch, mid, k=1),
            conv_norm_act(mid, mid, k=kernel, s=stride, groups=mid),
            SE3D(mid),
            nn.Conv3d(mid, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else F.silu(y)


class LocalMobileBranch(nn.Module):
    def __init__(self, base: int, out_ch: int):
        super().__init__()
        self.stem = conv_norm_act(1, base, 3, 2)
        self.b1 = MBConv3D(base, base * 2, stride=2, expand=2)
        self.b2 = MBConv3D(base * 2, base * 4, stride=2, expand=2)
        self.b3 = MBConv3D(base * 4, base * 8, stride=2, expand=2, kernel=5)
        self.proj = conv_norm_act(base * 8, out_ch, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        mid = self.b1(x)
        x = self.b2(mid)
        x = self.b3(x)
        x = self.proj(x)
        return x, mid


class TinyTransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dropout: float):
        super().__init__()
        # nn.MultiheadAttention requires d_model % nhead == 0. Fail loudly here instead of
        # letting an inconsistent config surface later as a deep PyTorch assertion.
        if dim % heads:
            raise ValueError(
                f"token dimension {dim} must be divisible by attention heads {heads}"
            )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, tokens):
        return self.encoder(tokens)


class GlobalTransformerBranch(nn.Module):
    def __init__(self, base: int, out_ch: int, token_grid: int, depth: int, heads: int, dropout: float):
        super().__init__()
        self.token_grid = int(token_grid)
        self.patch_embed = nn.Sequential(
            conv_norm_act(1, base, 3, 2),
            conv_norm_act(base, base * 2, 3, 2),
            conv_norm_act(base * 2, out_ch, 3, 2),
        )
        self.encoder = TinyTransformerEncoder(out_ch, depth, heads, dropout)
        self.pos = nn.Parameter(torch.zeros(1, self.token_grid**3, out_ch))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.out = conv_norm_act(out_ch, out_ch, 1)

    def forward(self, x, out_size):
        feat = self.patch_embed(x)
        pooled = F.adaptive_avg_pool3d(feat, (self.token_grid, self.token_grid, self.token_grid))
        b, c, d, h, w = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens + self.pos[:, : tokens.shape[1]])
        feat = tokens.transpose(1, 2).reshape(b, c, d, h, w)
        feat = F.interpolate(feat, size=out_size, mode="trilinear", align_corners=False)
        return self.out(feat)


class MorphGateFusion(nn.Module):
    def __init__(self, local_ch: int, global_ch: int, out_ch: int, fixed: bool = False, use_morphology: bool = True):
        super().__init__()
        self.fixed = fixed
        self.use_morphology = use_morphology
        self.local_proj = conv_norm_act(local_ch, out_ch, 1)
        self.global_proj = conv_norm_act(global_ch, out_ch, 1)
        gate_in_ch = out_ch * 2 + (2 if use_morphology else 0)
        self.gate = nn.Sequential(
            conv_norm_act(gate_in_ch, out_ch // 2, 3),
            nn.Conv3d(out_ch // 2, 1, 1),
        )
        self.post = nn.Sequential(conv_norm_act(out_ch, out_ch, 3), SE3D(out_ch))

    def morphology_maps(self, local, global_):
        density = torch.mean(torch.abs(local), dim=1, keepdim=True)
        diff = torch.mean(torch.abs(local - global_), dim=1, keepdim=True)
        return density, diff

    def forward(self, local, global_):
        local = self.local_proj(local)
        global_ = self.global_proj(global_)
        if self.fixed:
            weight = torch.full_like(local[:, :1], 0.5)
        else:
            gate_inputs = [local, global_]
            if self.use_morphology:
                density, diff = self.morphology_maps(local, global_)
                gate_inputs.extend([density, diff])
            weight = torch.sigmoid(self.gate(torch.cat(gate_inputs, dim=1)))
        fused = weight * local + (1.0 - weight) * global_
        smooth = torch.mean(torch.abs(weight[:, :, 1:] - weight[:, :, :-1]))
        smooth = smooth + torch.mean(torch.abs(weight[:, :, :, 1:] - weight[:, :, :, :-1]))
        smooth = smooth + torch.mean(torch.abs(weight[:, :, :, :, 1:] - weight[:, :, :, :, :-1]))
        return self.post(fused), weight, smooth


class DetectorHead(nn.Module):
    def __init__(self, in_ch: int, dropout: float, pool_mode: str = "global", center_frac: float = 0.25):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.pool_mode = str(pool_mode)
        self.center_frac = float(center_frac)
        hidden = max(64, in_ch // 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_ch),
            nn.Dropout(dropout),
            nn.Linear(in_ch, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(hidden, 1)

    def forward(self, x):
        # pool_mode="center": this is [candidate-level] classification -- the candidate point sits
        # exactly at the patch center, while a nodule occupies only 0.02%~0.5% of the 96³ volume.
        # The original implementation used global average pooling (AdaptiveAvgPool3d(1)), which
        # dilutes the central nodule's features by a factor of thousands, effectively asking the
        # classifier to look at "the average appearance of the whole patch". Pooling the central
        # region instead is what makes the model consistent with the "nodule is at the center" prior.
        # The default is still global, so compatibility with existing configs/weights is exact
        # (pool_mode is not a parameter, so the state_dict is unchanged).
        if self.pool_mode == "center":
            d, h, w = x.shape[-3:]
            cd = max(1, int(round(d * self.center_frac)))
            ch = max(1, int(round(h * self.center_frac)))
            cw = max(1, int(round(w * self.center_frac)))
            z0 = max(0, (d - cd) // 2)
            y0 = max(0, (h - ch) // 2)
            x0 = max(0, (w - cw) // 2)
            x = x[..., z0 : z0 + cd, y0 : y0 + ch, x0 : x0 + cw]
        h = self.fc(self.pool(x))
        return {"logit": self.cls(h).squeeze(1)}


class SAFFNoduleDet(nn.Module):
    def __init__(self, cfg: dict[str, Any], variant: str = "full"):
        super().__init__()
        if variant not in SUPPORTED_SAFF_VARIANTS:
            allowed = ", ".join(SUPPORTED_SAFF_VARIANTS)
            raise ValueError(f"Unsupported SAFF ablation variant: {variant}. Allowed: {allowed}")
        mcfg = cfg["model"]
        base = int(mcfg["base_channels"])
        local_ch = int(mcfg["local_channels"])
        global_ch = int(mcfg["global_channels"])
        fusion_ch = int(mcfg["fusion_channels"])
        token_grid = int(mcfg["global_token_grid"])
        if variant == "full_attention":
            token_grid = int(mcfg.get("full_attention_token_grid", token_grid))
        self.variant = variant
        self.use_local = variant != "transformer_only"
        self.use_global = variant != "cnn_only"
        self.use_distill = variant != "no_distill"
        self.local = LocalMobileBranch(base, local_ch) if self.use_local else None
        self.global_branch = (
            GlobalTransformerBranch(
                base,
                global_ch,
                token_grid=token_grid,
                depth=int(mcfg["transformer_depth"]),
                heads=int(mcfg["transformer_heads"]),
                dropout=float(mcfg["dropout"]),
            )
            if self.use_global
            else None
        )
        if self.use_local and self.use_global:
            self.fusion = MorphGateFusion(
                local_ch,
                global_ch,
                fusion_ch,
                fixed=variant == "fixed_fusion",
                use_morphology=variant != "no_morph_gate",
            )
            head_ch = fusion_ch
        elif self.use_local:
            self.local_only_proj = conv_norm_act(local_ch, fusion_ch, 1)
            head_ch = fusion_ch
        else:
            self.global_only_proj = conv_norm_act(global_ch, fusion_ch, 1)
            head_ch = fusion_ch
        self.head = DetectorHead(head_ch, float(mcfg["dropout"]),
                                  str(mcfg.get("head_pool", "global")),
                                  float(mcfg.get("head_center_frac", 0.25)))
        self.local_distill = nn.Conv3d(local_ch, fusion_ch, 1) if self.use_local else None
        self.global_distill = nn.Conv3d(global_ch, fusion_ch, 1) if self.use_global else None

    def forward(self, x):
        extras = {}
        local_feat = None
        global_feat = None
        if self.use_local:
            local_feat, _ = self.local(x)
        if self.use_global:
            out_size = local_feat.shape[-3:] if local_feat is not None else (6, 6, 6)
            global_feat = self.global_branch(x, out_size=out_size)
        if local_feat is not None and global_feat is not None:
            fused, gate, smooth = self.fusion(local_feat, global_feat)
            extras["gate_smooth"] = smooth
            extras["gate_mean"] = gate.mean()
            if self.use_distill:
                local_projected = F.normalize(self.local_distill(local_feat), dim=1)
                global_projected = F.normalize(self.global_distill(global_feat), dim=1)
                extras["distill_loss"] = F.mse_loss(local_projected, global_projected)
        elif local_feat is not None:
            fused = self.local_only_proj(local_feat)
        else:
            fused = self.global_only_proj(global_feat)
        out = self.head(fused)
        out.update(extras)
        return out


class UNet3DBaseline(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        base = int(cfg["model"]["base_channels"])
        self.enc1 = nn.Sequential(conv_norm_act(1, base), conv_norm_act(base, base))
        self.enc2 = nn.Sequential(conv_norm_act(base, base * 2, 3, 2), conv_norm_act(base * 2, base * 2))
        self.enc3 = nn.Sequential(conv_norm_act(base * 2, base * 4, 3, 2), conv_norm_act(base * 4, base * 4))
        self.enc4 = nn.Sequential(conv_norm_act(base * 4, base * 8, 3, 2), conv_norm_act(base * 8, base * 8))
        self.head = DetectorHead(base * 8, float(cfg["model"]["dropout"]),
                                  str(cfg["model"].get("head_pool", "global")),
                                  float(cfg["model"].get("head_center_frac", 0.25)))  # [headpool-fix]

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return self.head(x)


class ConvNeXtBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.BatchNorm3d(channels)
        self.pw1 = nn.Conv3d(channels, channels * 4, 1)
        self.pw2 = nn.Conv3d(channels * 4, channels, 1)

    def forward(self, x):
        y = self.dw(x)
        y = self.norm(y)
        y = F.gelu(self.pw1(y))
        y = self.pw2(y)
        return x + y


class ConvNeXt3DBaseline(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        base = int(cfg["model"]["base_channels"])
        chans = [base, base * 2, base * 4, base * 8]
        layers = [conv_norm_act(1, chans[0], 3, 2), ConvNeXtBlock3D(chans[0])]
        for i in range(1, len(chans)):
            layers += [conv_norm_act(chans[i - 1], chans[i], 3, 2), ConvNeXtBlock3D(chans[i])]
        self.net = nn.Sequential(*layers)
        self.head = DetectorHead(chans[-1], float(cfg["model"]["dropout"]),
                                  str(cfg["model"].get("head_pool", "global")),
                                  float(cfg["model"].get("head_center_frac", 0.25)))  # [headpool-fix]

    def forward(self, x):
        return self.head(self.net(x))


class SwinUNETRAdapted(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        mcfg = cfg["model"]
        base = int(mcfg["base_channels"])
        ch = int(mcfg["fusion_channels"])
        self.stem = nn.Sequential(conv_norm_act(1, base, 3, 2), conv_norm_act(base, ch, 3, 2))
        self.token_grid = int(mcfg["global_token_grid"])
        # The Swin-adapted baseline uses a 256-d token, which is not divisible by the
        # default transformer_heads=6; it has its own head count (default 8).
        heads = int(mcfg.get("swin_adapted_heads", 8))
        self.encoder = TinyTransformerEncoder(ch, int(mcfg["transformer_depth"]), heads, float(mcfg["dropout"]))
        self.pos = nn.Parameter(torch.zeros(1, self.token_grid**3, ch))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.post = nn.Sequential(conv_norm_act(ch, ch, 3), SE3D(ch))
        self.head = DetectorHead(ch, float(mcfg["dropout"]),
                                  str(mcfg.get("head_pool", "global")),
                                  float(mcfg.get("head_center_frac", 0.25)))  # [headpool-fix]

    def forward(self, x):
        feat = self.stem(x)
        pooled = F.adaptive_avg_pool3d(feat, (self.token_grid, self.token_grid, self.token_grid))
        b, c, d, h, w = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens + self.pos[:, : tokens.shape[1]])
        y = tokens.transpose(1, 2).reshape(b, c, d, h, w)
        y = F.interpolate(y, size=feat.shape[-3:], mode="trilinear", align_corners=False)
        return self.head(self.post(y + feat))


class CoTr3DBaseline(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        mcfg = cfg["model"]
        base = int(mcfg["base_channels"])
        self.cnn = LocalMobileBranch(base, int(mcfg["local_channels"]))
        self.trans = TinyTransformerEncoder(
            int(mcfg["local_channels"]),
            max(1, int(mcfg["transformer_depth"]) - 1),
            int(mcfg["transformer_heads"]),
            float(mcfg["dropout"]),
        )
        self.head = DetectorHead(int(mcfg["local_channels"]), float(mcfg["dropout"]),
                                  str(mcfg.get("head_pool", "global")),
                                  float(mcfg.get("head_center_frac", 0.25)))  # [headpool-fix]

    def forward(self, x):
        feat, _ = self.cnn(x)
        b, c, d, h, w = feat.shape
        pooled = F.adaptive_avg_pool3d(feat, (4, 4, 4))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.trans(tokens)
        y = tokens.transpose(1, 2).reshape(b, c, 4, 4, 4)
        y = F.interpolate(y, size=(d, h, w), mode="trilinear", align_corners=False)
        return self.head(feat + y)


def build_model(name: str, cfg: dict[str, Any], variant: str = "full") -> nn.Module:
    if name == "saff_noduledet":
        return SAFFNoduleDet(cfg, variant=variant)
    if name == "unet3d":
        return UNet3DBaseline(cfg)
    if name == "convnext3d":
        return ConvNeXt3DBaseline(cfg)
    if name == "swin_unetr":
        return SwinUNETRAdapted(cfg)
    if name == "cotr3d":
        return CoTr3DBaseline(cfg)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
