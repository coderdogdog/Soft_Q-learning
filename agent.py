# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from net import SoftQNetwork, PolicyNetwork
from config import DEVICE, STATE_DIM, ACTION_DIM, HIDDEN_DIM, \
    ALPHA, N_PARTICLES, BATCH_SIZE, GAMMA, LR_Q, LR_POLICY, TAU


class ReplayBuffer:
    def __init__(self, max_len: int, state_dim: int, action_dim: int):
        self.max_len = max_len

        self.next_idx = 0           # 下一个要写入的位置
        self.count = 0              # 当前已有数据量

        # 为每个数据字段预分配 NumPy 数组
        self.states = np.zeros((max_len, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_len, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_len, 1), dtype=np.float32)
        self.next_states = np.zeros((max_len, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_len, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        idx = self.next_idx
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.count = min(self.count + 1, self.max_len)
        self.next_idx = (self.next_idx + 1) % self.max_len

    def sample(self, batch_size):
        """随机采样一个 batch"""
        # 从 [0, self.count) 范围内随机选取 batch_size 个索引
        # replace=False 的意思是：不放回抽样 不抽一样的
        indices = np.random.choice(self.count, batch_size, replace=False)
        # 直接从数组中根据索引取值
        return (self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices])


class SoftQLearningAgent:

    def __init__(self):
        # Q网络（使用Double Q技巧[reference:15]）
        self.q_net1 = SoftQNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
        self.target_q_net1 = SoftQNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())

        self.q_net2 = SoftQNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
        self.target_q_net2 = SoftQNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # 策略网络
        self.policy_net = PolicyNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)

        # 优化器
        self.q_optimizer = optim.Adam(
            list(self.q_net1.parameters()) + list(self.q_net2.parameters()),
            lr=LR_Q
        )
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LR_POLICY)

        # 目标网络软更新系数
        self.tau = TAU

        # 训练步数计数
        self.update_counter = 0

    def select_action(self, state, explore=True):
        """选择动作（用于与环境交互）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if explore:
                # 训练模式 收集数据
                action, _ = self.policy_net.sample(state, n_particles=1)
            else:
                # 评估模式：使用均值动作（确定性）
                mean, _ = self.policy_net.forward(state)
                action = torch.tanh(mean)

        return action.cpu().numpy()[0]

    def _compute_soft_value(self, q_values):
        """
        计算 Soft Value Function:
        V_soft(s) = alpha * log( ∫ exp( Q_soft(s, a) / alpha ) da )
        使用 log-sum-exp 近似积分

        输入 q_values       [batch, particles] 
        输出 soft_values     [batch, 1]
        """
        soft_values = ALPHA * torch.logsumexp(q_values / ALPHA, dim=1, keepdim=True)
        return soft_values

    def _compute_svgd_gradient(self, states, actions, q_values):
        """
        这里不需要迭代多次，只需要算 “方向” ，迭代可以依靠外层的网络参数更新推进
        这就是: Soft Q-Learning 采用的摊销（Amortized SVGD） 技巧

        计算SVGD更新方向:
        Δf(s) = E_{a'~π}[ κ(a, a') ∇_{a'} Q(s, a') + α ∇_{a'} κ(a, a') ]

        参数:
            states: [batch, state_dim]
            actions: [batch, n_particles, action_dim]
            q_values: [batch, n_particles]
        """
        batch_size, n_particles, action_dim = actions.shape

        # 展平以便计算
        states_flat = states.unsqueeze(1).expand(-1, n_particles, -1).reshape(-1, STATE_DIM)    # 转为 形状 [b*n, STATE_DIM]
        actions_flat = actions.reshape(-1, action_dim)      # 转为 形状 [b*n, ACTION_DIM]
        q_flat = q_values.reshape(-1, 1)                    # 转为 形状 [b*n, 1]

        # 计算核函数 κ(a, a')（RBF核）
        # 这里 a 是当前粒子，a' 是所有粒子
        actions_flat.requires_grad_(True)

        # 计算所有粒子对的距离矩阵
        # [n_total, 1, action_dim] - [1, n_total, action_dim]
        diff = actions_flat.unsqueeze(1) - actions_flat.unsqueeze(0)    # [n_total, n_total, action_dim]
        sq_dist = (diff ** 2).sum(dim=-1)       # [n_total, n_total]
        """        
        核带宽代码中常用0.5 
        带宽太大：排斥力太弱，粒子容易模式坍塌（Mode Collapse），只覆盖一个峰值。
        带宽太小：排斥力太强，粒子可能振荡不稳定，或无法有效探索。
        """
        bandwidth = 0.5  # 核带宽 
        kernel = torch.exp(-sq_dist / (2 * bandwidth ** 2))  # [n_total, n_total]

        # 计算 ∇_{a'} Q(s, a')
        q_flat_sum = q_flat.sum()
        grad_q = torch.autograd.grad(
            q_flat_sum, actions_flat, create_graph=False, retain_graph=False
        )[0]  # [n_total, action_dim]

        # 计算 ∇_{a'} κ(a, a')
        # 对 a' 求梯度: d_kernel/da' = (a - a') / bandwidth^2 * kernel
        grad_kernel = -(diff / (bandwidth ** 2)) * kernel.unsqueeze(-1)  # [n_total, n_total, action_dim]

        # 计算 SVGD 方向: (1/n) * Σ [ κ(a, a') ∇_{a'} Q + α ∇_{a'} κ ]
        # 对每个粒子 i: Δf_i = (1/n) Σ_j [ κ(a_i, a_j) ∇_{a_j} Q + α ∇_{a_j} κ(a_i, a_j) ]
        # 这里 α 是温度参数（与熵权重共享）
        n_total = actions_flat.shape[0]

        # 第一项: κ * ∇Q
        term1 = (kernel.unsqueeze(-1) * grad_q.unsqueeze(0)).sum(dim=1) / n_total  # [n_total, action_dim]

        # 第二项: α * ∇κ
        term2 = ALPHA * grad_kernel.sum(dim=1) / n_total  # [n_total, action_dim]

        delta_f = term1 + term2  # [n_total, action_dim]

        # 恢复形状 [batch, n_particles, action_dim]
        delta_f = delta_f.reshape(batch_size, n_particles, action_dim)

        return delta_f.detach()  # 停止梯度，作为回归目标

    def update(self, replay_buffer, batch_size, writer):
        """执行一次更新：Q学习 + SVGD策略改进"""

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        # 转换为 PyTorch Tensor
        states = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(action, dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(reward, dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(done, dtype=torch.float32).to(DEVICE)

        # 更新 Q 网络
        with torch.no_grad():
            # 从当前策略采样 next_actions（多个粒子）
            next_actions, _ = self.policy_net.sample(next_states, n_particles=N_PARTICLES)
            # [batch, n_particles, action_dim]

            # 计算目标 Q 值: Q_target(s', a') = min(Q1, Q2) 使用 Double Q 技巧
            next_q1 = self.target_q_net1(
                next_states.unsqueeze(1).expand(-1, N_PARTICLES, -1).reshape(-1, STATE_DIM),
                next_actions.reshape(-1, ACTION_DIM)
            ).reshape(BATCH_SIZE, N_PARTICLES)

            next_q2 = self.target_q_net2(
                next_states.unsqueeze(1).expand(-1, N_PARTICLES, -1).reshape(-1, STATE_DIM),
                next_actions.reshape(-1, ACTION_DIM)
            ).reshape(BATCH_SIZE, N_PARTICLES)

            next_q = torch.min(next_q1, next_q2)  # [batch, n_particles]

            # 计算 Soft Value: V_soft(s') = alpha * logsumexp( Q(s', a') / alpha )
            next_v = self._compute_soft_value(next_q)  # [batch, 1]

            # 计算目标: Q_target = r + gamma * (1-done) * V_soft(s')
            target_q = rewards + GAMMA * (1 - dones) * next_v

        # 当前 Q 值
        current_q1 = self.q_net1(states, actions)
        current_q2 = self.q_net2(states, actions)

        # Q 损失
        q1_loss = F.smooth_l1_loss(current_q1, target_q.detach(), reduction='mean')
        q2_loss = F.smooth_l1_loss(current_q2, target_q.detach(), reduction='mean')
        # q1_loss = F.mse_loss(current_q1, target_q.detach())
        # q2_loss = F.mse_loss(current_q2, target_q.detach())
        q_loss = q1_loss + q2_loss

        # 更新 Q 网络
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), 0.5)
        self.q_optimizer.step()

        # 使用 SVGD 更新策略网络
        # 从当前策略采样动作粒子
        sampled_actions, _ = self.policy_net.sample(states, n_particles=N_PARTICLES)
        # [batch, n_particles, action_dim]

        # 计算每个粒子的 Q 值
        states_expanded = states.unsqueeze(1).expand(-1, N_PARTICLES, -1).reshape(-1, STATE_DIM)
        actions_flat = sampled_actions.reshape(-1, ACTION_DIM)

        q1_particles = self.q_net1(states_expanded, actions_flat).reshape(BATCH_SIZE, N_PARTICLES)
        q2_particles = self.q_net2(states_expanded, actions_flat).reshape(BATCH_SIZE, N_PARTICLES)
        q_particles = torch.min(q1_particles, q2_particles)  # [batch, n_particles]

        # 计算 SVGD 更新方向
        delta_f = self._compute_svgd_gradient(states, sampled_actions, q_particles)
        # [batch, n_particles, action_dim]

        # 策略损失：让网络输出的动作向 delta_f 方向移动
        # 等价于最小化 - E[ a^T * delta_f ]
        policy_loss = -(
                sampled_actions * delta_f
        ).sum(dim=-1).mean()

        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()

        # 软更新目标网络
        self.update_counter += 1
        if self.update_counter % 2 == 0:  # 每2步更新一次目标网络
            for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 控制记录频率
        if self.update_counter % 10 == 0:
            # TensorBoard 记录
            writer.add_scalar("train/q_loss", q_loss.item(), self.update_counter)
            writer.add_scalar("train/policy_loss", policy_loss.item(), self.update_counter)

    def save(self, path_policy_net):
        """保存策略网络权重"""
        torch.save(self.policy_net.state_dict(), path_policy_net)

    def load(self, path_policy_net):
        """加载策略网络权重"""
        self.policy_net.load_state_dict(torch.load(path_policy_net, map_location=DEVICE, weights_only=True))
