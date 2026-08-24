# -*- coding: utf-8 -*-
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from agent import ReplayBuffer, SoftQLearningAgent
from utils import evaluate_agent, set_seed

from config import DEVICE, STATE_DIM, ACTION_DIM, HIDDEN_DIM, \
    ALPHA, N_PARTICLES, BATCH_SIZE, GAMMA, LR_Q, LR_POLICY, TAU,\
    ENV_NAME


max_env_steps = 1000000
warmup_steps = 10000

# =================== 训练 =======================
def train():
    # 模型文件夹 =================================
    # ./model/env_name/
    model_dir = f"./model/{ENV_NAME}/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # 设置日志文件夹 =================================
    # 训练日志
    log_dir = f'''runs/{ENV_NAME}/TD3_{datetime.now().strftime("%Y%m%d_%H_%M_%S")}'''
    writer = SummaryWriter(log_dir)
    print(f"在新的终端窗口运行TensorBoard: tensorboard --logdir=runs/{ENV_NAME}")

    # 创建训练环境 =================================
    env = gym.make(ENV_NAME, render_mode=None)
    # 创建用来测试评估的环境
    evaluate_env = gym.make(ENV_NAME, render_mode=None)

    # 设置随机种子 =================================
    set_seed(20)
    env.reset(seed=20)

    evaluate_env.reset(seed=30)

    # 设置智能体 =================================
    sql_agent = SoftQLearningAgent()

    # 设置经验池 =================================
    buffer = ReplayBuffer(max_len=1000000, state_dim=STATE_DIM, action_dim=ACTION_DIM)

    # 初始化 =================================
    state, _ = env.reset()
    total_steps = 0
    print("开始训练！")
    while total_steps < max_env_steps:
        if total_steps < warmup_steps:
            action = env.action_space.sample()  # 预热期随机探索
        else:
            action = sql_agent.select_action(state)

        next_state, reward, terminated, truncated, infos = env.step(action)
        done = terminated or truncated

        buffer.store(state, action, reward, next_state, terminated)
        total_steps += 1

        if done:
            state, _ = env.reset()
        else:
            state = next_state

        if total_steps > warmup_steps:
            sql_agent.update(buffer, BATCH_SIZE, writer)

            if sql_agent.update_counter % 1000 == 0:
                avg_r, avg_steps = evaluate_agent(evaluate_env, sql_agent,
                                                    is_render_human=False, test_numb=3)

                print(f"游戏总步数: {total_steps:8d} | 网络更新次数: {sql_agent.update_counter:8d} | "
                        f"每回合平均奖励: {avg_r:9.2f} | 每回合平均步数: {avg_steps:5d}")

                writer.add_scalar("test/avg_r", avg_r, sql_agent.update_counter)
                writer.add_scalar("test/avg_steps", avg_steps, sql_agent.update_counter)

            # 保存策略网络
            if sql_agent.update_counter % 5000 == 0:
                path_name = "train_" + str(sql_agent.update_counter) + ".pth"
                save_path = model_dir + path_name
                sql_agent.save(save_path)

    env.close()
    evaluate_env.close()
    writer.close()


if __name__ == "__main__":
    train()

