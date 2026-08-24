# -*- coding: utf-8 -*-
import torch
import gymnasium as gym
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from agent import ReplayBuffer, SoftQLearningAgent
from utils import evaluate_agent, set_seed

from config import DEVICE, STATE_DIM, ACTION_DIM, HIDDEN_DIM, \
    ALPHA, N_PARTICLES, BATCH_SIZE, GAMMA, LR_Q, LR_POLICY, TAU, \
    ENV_NAME, ACTION_BOUND


load_model_name = input("请输入模型文件名(.pth)：")
test_num = input("请输入测试环境几局：")
test_num = int(test_num)

# 创建环境
env_name = ENV_NAME
test_env = gym.make(env_name, render_mode="human")

# 测试环境随机种子
test_env.reset(seed=50)

state_dim = STATE_DIM
action_dim = ACTION_DIM
action_bound = ACTION_BOUND

# 创建智能体
sql_agent = SoftQLearningAgent()

load_path = f"./model/{env_name}/" + load_model_name
Path(load_path).parent.mkdir(parents=True, exist_ok=True)

sql_agent.load(load_path)

is_render_human = True

avg_scores, avg_steps = evaluate_agent(test_env, sql_agent, is_render_human, test_num)
test_env.close()
print(f"======= 平均每局得分：{avg_scores:8.2f} ======= ")
print(f"======= 平均每局步数：{avg_steps:8d} ======= ")
