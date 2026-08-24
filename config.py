# -*- coding: utf-8 -*-

# ==================== 环境配置 ====================
ENV_NAME = 'BipedalWalker-v3'  # 或 'BipedalWalker-v2'[reference:5]
STATE_DIM = 24  # 观测空间维度[reference:6]
ACTION_DIM = 4  # 动作空间维度[reference:7]
ACTION_BOUND = 1.0  # 动作范围 [-1, 1]

# ==================== 超参数 ====================
DEVICE = "cuda:0"       # "cpu"
GAMMA = 0.99
ALPHA = 0.2  # 温度参数，控制熵的权重[reference:8]
LR_Q = 3e-4  # Q网络学习率
LR_POLICY = 3e-4  # 策略网络学习率
BATCH_SIZE = 256  # [reference:9]
REPLAY_SIZE = int(1e6)  # [reference:10][reference:11]
TAU = 0.005  # 目标网络软更新系数[reference:12]
N_PARTICLES = 20  # SVGD粒子数量（每个状态采样的动作数）
HIDDEN_DIM = 256  # [reference:13]
