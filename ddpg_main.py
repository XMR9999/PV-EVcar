import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
import subprocess
import sys

# --- 步骤 0: 自动安装依赖 ---
try:
    print("正在检查并安装必要的Python库 (torch)...")
    # 注意: 这里安装的是CPU版本的PyTorch。如果您有GPU，请访问PyTorch官网获取对应CUDA版本的安装命令。
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    print("库已成功安装或已存在。")
except subprocess.CalledProcessError as e:
    print(f"在安装库时发生错误: {e}")
    print("请手动运行 'pip install torch' 后再试。")
    sys.exit(1)

# 导入我们之前创建的环境
from ev_env import MultiRegionEnv

# --- 1. 经验回放池 (Replay Buffer) ---
class ReplayBuffer:
    """一个固定大小的经验回放池，用于存储和采样经验元组。"""
    def __init__(self, buffer_size, batch_size):
        self.memory = collections.deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        """将一个经验元组存入记忆库。"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """从记忆库中随机采样一批经验。"""
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

# --- 2. 演员网络 (Actor Network) ---
class Actor(nn.Module):
    """演员网络：输入状态，输出确定的动作。"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        # 使用tanh将输出限制在[-1, 1]之间，然后乘以最大动作值
        return self.max_action * torch.tanh(self.layer_3(x))

# --- 3. 评论家网络 (Critic Network) ---
class Critic(nn.Module):
    """评论家网络：输入状态和动作，输出Q值。"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 1)

    def forward(self, state, action):
        # 将状态和动作拼接在一起作为输入
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

# --- 4. DDPG智能体 (Agent) ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        # 初始化网络
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 初始化经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size=100000, batch_size=100)
        
        # 定义超参数
        self.max_action = max_action
        self.discount = 0.99  # 折扣因子 gamma
        self.tau = 0.005      # 目标网络软更新系数

    def select_action(self, state):
        """根据当前状态选择动作。"""
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        """从经验池中采样并更新网络。"""
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return

        # 1. 采样
        batch = self.replay_buffer.sample()
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done, dtype=np.float32)).unsqueeze(1)

        # 2. 更新Critic网络
        # 计算目标Q值
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1 - done) * self.discount * target_Q
        
        # 计算当前Q值
        current_Q = self.critic(state, action)
        
        # 计算Critic损失
        critic_loss = F.mse_loss(current_Q, target_Q.detach())
        
        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # --- 代码修正处: 添加梯度裁剪 ---
        # 在优化器更新权重之前，对梯度进行裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        
        self.critic_optimizer.step()

        # 3. 更新Actor网络
        # 计算Actor损失
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # 优化Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        """保存模型权重。"""
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")

# --- 5. 训练主循环 ---
if __name__ == '__main__':
    # --- 初始化 ---
    # 定义文件路径和参数
    DATA_FILES = {
        'pv_a': "One year of solar photovoltaic data A.csv",
        'pv_c': "One year of solar photovoltaic data C.csv",
        'charging_load': "charging load.csv",
        'residential_load': "residential load.csv",
    }
    
    # 从您的get_parameters函数获取参数
    # (这里为了脚本独立性，简化了参数)
    PARAMS = {
        'T': 96, 'delta_t': 0.25, 'regions': ['A', 'B', 'C'],
        'price_buy': {t: 0.25 if 0 <= t * 0.25 < 7 else (1.2 if (10 <= t * 0.25 < 12) or (18 <= t * 0.25 < 21) else 0.8) for t in range(96)},
        'price_sell': 0.4, 'cost_curtailment': 0.1, 'cost_ess_deg': 0.08, 
        'cost_transmission': 0.02, 'trans_loss_factor': 0.05, 'trans_max_power': 200,
        'ess_capacity': {'A': 450, 'B': 600, 'C': 350}, 'ess_soc_min': 0.2, 
        'ess_soc_max': 0.95, 'ess_charge_eff': 0.95, 'ess_discharge_eff': 0.95,
        'ess_p_charge_max': {'A': 150, 'B': 120, 'C': 100}, 
        'ess_p_discharge_max': {'A': 150, 'B': 120, 'C': 100},
        'ess_initial_soc_val': 0.5
    }

    # 创建环境和智能体
    env = MultiRegionEnv(data_files=DATA_FILES, params=PARAMS)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = DDPGAgent(state_dim, action_dim, max_action)

    # 训练参数
    max_episodes = 500  # 总训练回合数
    max_timesteps = 96  # 每个回合的最大步数
    expl_noise = 0.1    # 探索噪声的标准差
    
    episode_rewards = [] # 记录每个回合的总奖励

    # --- 开始训练 ---
    print("\n--- 开始DDPG训练 ---")
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(max_timesteps):
            # 选择动作并添加噪声
            action = agent.select_action(state)
            noise = np.random.normal(0, max_action * expl_noise, size=action_dim)
            action = (action + noise).clip(env.action_space.low, env.action_space.high)
            
            # 与环境交互
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存入经验池
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新网络
            agent.update()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode: {episode+1}, Avg. Reward: {avg_reward:.2f}")

    # --- 训练结束 ---
    print("--- 训练完成 ---")
    agent.save("ddpg_model")
    print("模型已保存为 'ddpg_model_actor.pth' 和 'ddpg_model_critic.pth'")

    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title("DDPG Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("ddpg_learning_curve.png")
    plt.show()
    print("学习曲线图已保存为 'ddpg_learning_curve.png'")
    
    env.close()

