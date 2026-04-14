import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=5, n_groups=4, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.gn1 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.gn2 = nn.GroupNorm(n_groups, out_channels)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.mish = nn.Mish()

    def forward(self, x, cond):
        residual = self.residual(x)
        bias = self.cond_proj(cond).unsqueeze(-1)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.mish(out + bias)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.mish(out)
        out = self.dropout(out)
        return out + residual


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        action_chunk=4,
        cond_dim=64,
        time_dim=128,
        down_dims=None,
        kernel_size=5,
        n_groups=4,
        dropout=0.1,
    ):
        super().__init__()
        if down_dims is None:
            down_dims = [64, 128]

        self.act_dim = act_dim
        self.action_chunk = action_chunk

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, time_dim),
        )

        self.input_proj = nn.Conv1d(act_dim, down_dims[0], 1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        in_ch = down_dims[0]
        for i, out_ch in enumerate(down_dims):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(in_ch, out_ch, time_dim, kernel_size, n_groups, dropout),
                        ConditionalResidualBlock1D(out_ch, out_ch, time_dim, kernel_size, n_groups, dropout),
                    ]
                )
            )
            self.downsample.append(
                nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
                if i < len(down_dims) - 1
                else nn.Identity()
            )
            in_ch = out_ch

        mid_ch = down_dims[-1]
        self.mid1 = ConditionalResidualBlock1D(mid_ch, mid_ch, time_dim, kernel_size, n_groups, dropout)
        self.mid2 = ConditionalResidualBlock1D(mid_ch, mid_ch, time_dim, kernel_size, n_groups, dropout)

        rev_dims = list(reversed(down_dims))
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(len(rev_dims) - 1):
            in_ch = rev_dims[i]
            out_ch = rev_dims[i + 1]
            skip_ch = out_ch
            self.upsample.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(out_ch + skip_ch, out_ch, time_dim, kernel_size, n_groups, dropout),
                        ConditionalResidualBlock1D(out_ch, out_ch, time_dim, kernel_size, n_groups, dropout),
                    ]
                )
            )

        self.output_proj = nn.Sequential(
            nn.Conv1d(down_dims[0], down_dims[0], 1),
            nn.Mish(),
            nn.Conv1d(down_dims[0], act_dim, 1),
        )

    def forward(self, noisy_action_chunk, timestep, obs_condition):
        cond = self.time_mlp(timestep) + self.obs_encoder(obs_condition)
        x = self.input_proj(noisy_action_chunk.permute(0, 2, 1))

        skips = []
        for i, block in enumerate(self.down_blocks):
            x = block[0](x, cond)
            x = block[1](x, cond)
            skips.append(x)
            x = self.downsample[i](x)

        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        for i, block in enumerate(self.up_blocks):
            x = self.upsample[i](x)
            skip = skips[-(i + 2)]
            if x.shape[-1] != skip.shape[-1]:
                x = nn.functional.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block[0](x, cond)
            x = block[1](x, cond)

        return self.output_proj(x).permute(0, 2, 1)
