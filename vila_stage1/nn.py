import math

import torch
import torch.nn as nn

from .utils import weight_init



class MLPBlock(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, expand * dim),
            nn.ReLU6(),
            nn.Linear(expand * dim, dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


class LatentActHead(nn.Module):
    def __init__(self, act_dim, emb_dim, hidden_dim, expand=4, dropout=0.0):
        super().__init__()
        #TODO : modify 
        # self.proj0 = nn.Linear(2 * emb_dim, hidden_dim)
        # self.proj1 = nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim)
        # self.proj2 = nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim)
        # self.proj_end = nn.Linear(hidden_dim, act_dim)
        self.proj0 = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.proj1 = nn.Sequential(
            nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.proj2 = nn.Sequential(
            nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.proj_end = nn.Linear(hidden_dim, act_dim)
        
        self.block0 = MLPBlock(hidden_dim, expand, dropout)
        self.block1 = MLPBlock(hidden_dim, expand, dropout)
        self.block2 = MLPBlock(hidden_dim, expand, dropout)

    def forward(self, obs_emb, next_obs_emb):
        x = self.block0(self.proj0(torch.concat([obs_emb, next_obs_emb], dim=-1)))
        x = self.block1(self.proj1(torch.concat([x, obs_emb, next_obs_emb], dim=-1)))
        x = self.block2(self.proj2(torch.concat([x, obs_emb, next_obs_emb], dim=-1)))
        x = self.proj_end(x)
        return x


class LatentObsHead(nn.Module):
    def __init__(self, act_dim, proj_dim, hidden_dim, expand=4, dropout=0.0):
        super().__init__()

        self.proj0 = nn.Linear(act_dim + proj_dim, hidden_dim)
        self.proj1 = nn.Linear(act_dim + hidden_dim, hidden_dim)
        self.proj2 = nn.Linear(act_dim + hidden_dim, hidden_dim)
        self.proj_end = nn.Linear(hidden_dim, proj_dim)

        self.block0 = MLPBlock(hidden_dim, expand, dropout)
        self.block1 = MLPBlock(hidden_dim, expand, dropout)
        self.block2 = MLPBlock(hidden_dim, expand, dropout)

    def forward(self, x, action):
        x = self.block0(self.proj0(torch.concat([x, action], dim=-1)))
        x = self.block1(self.proj1(torch.concat([x, action], dim=-1)))
        x = self.block2(self.proj2(torch.concat([x, action], dim=-1)))
        x = self.proj_end(x)
        return x


# inspired by:
# 1. https://github.com/schmidtdominik/LAPO/blob/main/lapo/models.py
# 2. https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2, dropout=0.0, downscale=True):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self._downscale = downscale
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
            stride=2 if self._downscale else 1,
        )
        # conv downsampling is faster that maxpool, with same perf
        # self.conv = nn.Conv2d(
        #     in_channels=self._input_shape[0],
        #     out_channels=self._out_channels,
        #     kernel_size=3,
        #     padding=1,
        # )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels, dropout) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.conv(x)
        # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.blocks(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        if self._downscale:
            return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self._out_channels, h, w)


class Actor(nn.Module):
    def __init__(
        self,
        shape,
        num_actions,
        encoder_scale=1,
        encoder_channels=(16, 32, 32),
        encoder_num_res_blocks=1,
        dropout=0.0,
    ):
        super().__init__()
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.final_encoder_shape = shape
        self.encoder = nn.Sequential(
            *conv_stack,
            # nn.Flatten(),
        )
        self.actor_mean = nn.Sequential(
            nn.ReLU6(),
            # works either way...
            # nn.Linear(math.prod(shape), num_actions),
            nn.Linear(shape[0], num_actions),
        )
        self.num_actions = num_actions
        self.apply(weight_init)

    def forward(self, obs):
        out = self.encoder(obs)
        out = out.flatten(2).mean(-1)
        act = self.actor_mean(out)
        return act, out


class LAOMWithLabels(nn.Module):
    def __init__(
        self,
        shape,
        true_act_dim,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
        obs_head_dim=512,
        obs_head_dropout=0.0,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.LayerNorm(math.prod(shape), elementwise_affine=False) if encoder_norm_out else nn.Identity(),
        )
        self.idm_head = LatentActHead(latent_act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)
        self.true_actions_head = nn.Linear(latent_act_dim, true_act_dim)

        self.fdm_head = LatentObsHead(
            latent_act_dim, 
            math.prod(shape), 
            obs_head_dim, 
            dropout=obs_head_dropout,
        )

        self.final_encoder_shape = shape
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs, predict_true_act=False):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])

        latent_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))

        latent_next_obs = self.fdm_head(obs_emb.flatten(1).detach(), latent_action)
        # TODO: use norm from encoder here too!

        if predict_true_act:
            true_action = self.true_actions_head(latent_action)
            return latent_next_obs, latent_action, true_action, obs_emb.flatten(1).detach()

        return latent_next_obs, latent_action, obs_emb.flatten(1).detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        latent_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return latent_action
