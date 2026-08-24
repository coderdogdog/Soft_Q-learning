# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import DEVICE


LOG_2 = math.log(2)

def layer_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    """正交初始化线性层权重，偏置置零。"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


# Q网络（Soft Q-function）
class SoftQNetwork(nn.Module):
    """估计 Soft Q 值 Q_soft(s, a)"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # 状态 高维特征提取
        self.state_net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),  # 对隐藏层特征做归一化
            nn.GELU(),

            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),  # 第二层同样加归一化
            nn.GELU()
        )

        # 动作塔 (将低维动作映射到隐藏空间)
        # 为了保证输出到 hidden_dim 的特征方差稳定，建议使用gain = 1.0（因为不需要像 ReLU 那样放大方差）
        self.action_net = layer_init(nn.Linear(action_dim, hidden_dim), gain=1.0)

        # 融合
        # 状态特征(256) + 动作特征(256) = 512，再次经过归一化后输出Q值
        self.fusion_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.LayerNorm(hidden_dim),  # 融合后的特征也做归一化
            nn.GELU(),
            # 最后一层（输出层）：使用极小权重，初始 Q 值接近 0，稳定训练起步
            layer_init(nn.Linear(hidden_dim, 1), gain=0.01)  # 输出Q值，不加激活函数
        )

    def forward(self, state, action):
        state_feat = self.state_net(state)  # [batch, hidden_dim]
        action_feat = self.action_net(action)  # [batch, hidden_dim]
        combined = torch.cat([state_feat, action_feat], dim=-1)  # [batch, hidden_dim*2]
        q_value = self.fusion_net(combined)
        return q_value


# 策略网络（SVGD）
class PolicyNetwork(nn.Module):
    """
    策略网络: a = f^phi(s, xi)，xi ~ N(0, I)
    通过重参数化技巧采样，便于SVGD梯度传播
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            layer_init(nn.Linear(hidden_dim, action_dim)),
        )

        # 可学习的对数标准差参数    可以选择从0.0开始
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * 0.0)

    def forward(self, state):
        """返回动作的均值和log标准差"""
        mean = self.mean(state)

        # 裁剪，防止数值不稳定
        std = self.log_std.clamp(-20, 2).exp().expand_as(mean)
        return mean, std

    def sample(self, state, n_particles=1):
        """
        采样 n_particles 个动作
        返回: actions, log_probs
        """
        mean, std = self.forward(state)

        # 噪声 z 从标准正态分布采样
        normal_dist = torch.distributions.Normal(0, 1)

        """
        SQL的策略网络输出一种正态分布的均值 mean 和标准差 std 
        
        为什么使用重参数化技巧：假如直接用 mean std 直接构造 正态分布，动作需要通过采样才能得到，
        dist.sample() 采样会切断梯度传播，所以使用重参数化技巧：a = mean + std * z
        
        z从标准正态分布 N(0, I)中采样得到，a = mean + std * z 代表服从正态分布的动作粒子
        这些服从正态分布的动作粒子根据 SVGD 指引的方向，去拟合复杂的基于能量的策略分布
        同时z由采样得到，切断梯度传播，反向传播时，下游梯度只会流经 mean 和 std，去更新策略网络的参数。

        z 噪音 必须服从标准正态分布 N(0, I) 吗？
        数学上不必须，但是实际代码中经常用服从标准正态分布的噪音。
        因为其他正态分布可以由标准正态分布 平移（加μ）+ 缩放（乘 σ） 得到；
        另外在 SQL 和 SAC 中，必须通过计算动作的对数概率（Log Probability） 来计算熵，服从N(0, I)，对数概率方便计算。
        
        """
        # 重参数化采样
        if n_particles > 1:
            # [batch, action_dim] -> [batch, n_particles, action_dim]
            mean_exp = mean.unsqueeze(1).expand(-1, n_particles, -1)
            std_exp = std.unsqueeze(1).expand(-1, n_particles, -1)
            z = normal_dist.sample(mean_exp.shape).to(DEVICE)       # [batch, n_particles, action_dim]
            actions = mean_exp + std_exp * z
            # 计算log概率
            log_probs = normal_dist.log_prob(z).sum(dim=-1)         # [batch, n_particles]
            # 减去log_std的贡献（因为a = mean + std * z）
            log_probs = log_probs - torch.log(std_exp + 1e-8).sum(dim=-1)
        else:
            z = normal_dist.sample(mean.shape).to(DEVICE)
            actions = mean + std * z
            log_probs = normal_dist.log_prob(z).sum(dim=-1) - torch.log(std + 1e-8).sum(dim=-1)

        # 动作裁剪到 [-1, 1]
        actions = torch.tanh(actions)
        # 考虑tanh变换的log概率修正
        # correction = 2[log2 - x -Softplus(-2x) ]
        correction = 2 * (LOG_2 - actions - F.softplus(-2 * actions))
        log_probs = log_probs - correction.sum(dim=-1)

        return actions, log_probs

