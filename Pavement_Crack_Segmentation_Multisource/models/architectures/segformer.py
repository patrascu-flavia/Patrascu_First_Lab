# Public model architecture definition
# Extracted from the exact training implementation used in the associated study.
# Training, evaluation, plotting, and data-processing code are not included.

import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_dim,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, h, w


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim,
                dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, n, c = x.shape

        q = self.q(x)
        q = q.reshape(b, n, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_reduced = x.transpose(1, 2).reshape(b, c, h, w)
            x_reduced = self.sr(x_reduced)
            x_reduced = x_reduced.reshape(b, c, -1).transpose(1, 2)
            x_reduced = self.norm(x_reduced)
        else:
            x_reduced = x

        kv = self.kv(x_reduced)
        kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)

        k = kv[0]
        v = kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)

        return out


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
        )

    def forward(self, x, h, w):
        b, n, c = x.shape

        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        sr_ratio,
        mlp_ratio=4,
        dropout=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(
            dim=dim,
            hidden_dim=dim * mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)

        return x


class MixVisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=None,
        num_heads=None,
        depths=None,
        sr_ratios=None,
        mlp_ratio=4,
    ):
        super().__init__()

        if embed_dims is None:
            embed_dims = [32, 64, 160, 256]

        if num_heads is None:
            num_heads = [1, 2, 5, 8]

        if depths is None:
            depths = [2, 2, 2, 2]

        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]

        self.embed_dims = embed_dims

        self.patch_embed1 = OverlapPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dims[0],
            kernel_size=7,
            stride=4,
            padding=3,
        )

        self.patch_embed2 = OverlapPatchEmbed(
            in_channels=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.patch_embed3 = OverlapPatchEmbed(
            in_channels=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.patch_embed4 = OverlapPatchEmbed(
            in_channels=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.block1 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                sr_ratio=sr_ratios[0],
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depths[0])
        ])

        self.block2 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                sr_ratio=sr_ratios[1],
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depths[1])
        ])

        self.block3 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                sr_ratio=sr_ratios[2],
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depths[2])
        ])

        self.block4 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                sr_ratio=sr_ratios[3],
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depths[3])
        ])

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def tokens_to_feature(self, x, h, w):
        b, _, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)

        return x

    def forward(self, x):
        features = []

        x, h, w = self.patch_embed1(x)
        for block in self.block1:
            x = block(x, h, w)
        x = self.norm1(x)
        feature1 = self.tokens_to_feature(x, h, w)
        features.append(feature1)

        x, h, w = self.patch_embed2(feature1)
        for block in self.block2:
            x = block(x, h, w)
        x = self.norm2(x)
        feature2 = self.tokens_to_feature(x, h, w)
        features.append(feature2)

        x, h, w = self.patch_embed3(feature2)
        for block in self.block3:
            x = block(x, h, w)
        x = self.norm3(x)
        feature3 = self.tokens_to_feature(x, h, w)
        features.append(feature3)

        x, h, w = self.patch_embed4(feature3)
        for block in self.block4:
            x = block(x, h, w)
        x = self.norm4(x)
        feature4 = self.tokens_to_feature(x, h, w)
        features.append(feature4)

        return features


class SegFormerHead(nn.Module):
    def __init__(
        self,
        in_channels,
        decoder_dim=256,
        out_channels=1,
    ):
        super().__init__()

        self.linear_c1 = nn.Conv2d(in_channels[0], decoder_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(in_channels[1], decoder_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(in_channels[2], decoder_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(in_channels[3], decoder_dim, kernel_size=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(
                decoder_dim * 4,
                decoder_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Conv2d(
            decoder_dim,
            out_channels,
            kernel_size=1,
        )

    def forward(self, features, output_size):
        c1, c2, c3, c4 = features

        target_size = c1.shape[-2:]

        c1 = self.linear_c1(c1)

        c2 = self.linear_c2(c2)
        c2 = F.interpolate(
            c2,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        c3 = self.linear_c3(c3)
        c3 = F.interpolate(
            c3,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        c4 = self.linear_c4(c4)
        c4 = F.interpolate(
            c4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.fuse(x)
        x = self.classifier(x)

        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return x


class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        embed_dims=None,
        num_heads=None,
        depths=None,
        sr_ratios=None,
        mlp_ratio=4,
        decoder_dim=256,
    ):
        super().__init__()

        if embed_dims is None:
            embed_dims = [32, 64, 160, 256]

        self.encoder = MixVisionTransformerEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            sr_ratios=sr_ratios,
            mlp_ratio=mlp_ratio,
        )

        self.decode_head = SegFormerHead(
            in_channels=embed_dims,
            decoder_dim=decoder_dim,
            out_channels=out_channels,
        )

    def forward(self, x):
        output_size = x.shape[-2:]

        features = self.encoder(x)
        logits = self.decode_head(features, output_size)

        return logits


# ============================================================
# 7. Loss Functions
# ============================================================

