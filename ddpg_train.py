# -*- coding: utf-8 -*-
"""
DDPG Agent and Training Loop for Multi-Region EV Optimal Scheduling.

This script implements the DDPG agent and the main training process.
It imports the MultiRegionEVEnv class from the 'ev_env.py' file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- Import the custom environment ---
# Make sure 'ev_env.py' is in the same directory
from ev_env import MultiRegionEVEnv

# --- Hyperparameters ---
class HyperParams:
    def __init__(self):
        # --- MODIFICATIONS FOR PERFORMANCE TUNING ---
        # Reverted to the original baseline parameters that performed better,
        # but with increased training episodes to test for further improvement.
        self.replay_capacity = 100000
        self.lr_actor = 1e-4          # 恢复为原始的、效果更好的学习率
        self.lr_critic = 1e-3
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256
        self.hidden_dim = 256         # 恢复为原始的网络宽度
        self.noise_std = 0.1          # 恢复为原始的探索噪声
        self.max_episodes = 100      # 增加训练回合数，给予更长的学习时间
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

# --- Replay Buffer ---
class ReplayBuffer:
    """ A buffer for storing and sampling experience tuples. """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)

# --- Actor Network ---
class Actor(nn.Module):
    """ Policy network: maps state to action. """
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, s):
        s = F.relu(self.layer1(s))
        s = F.relu(self.layer2(s))
        # Use tanh to bound the action output to [-1, 1]
        a = torch.tanh(self.layer3(s)) * self.max_action
        return a

# --- Critic Network ---
class Critic(nn.Module):
    """ Value network: maps (state, action) to Q-value. """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        # Concatenate state and action
        sa = torch.cat([s, a], 1)
        q = F.relu(self.layer1(sa))
        q = F.relu(self.layer2(q))
        q = self.layer3(q)
        return q

# --- DDPG Agent ---
class DDPG:
    def __init__(self, state_dim, action_dim, max_action, params):
        self.params = params
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, params.hidden_dim, max_action).to(params.device)
        self.critic = Critic(state_dim, action_dim, params.hidden_dim).to(params.device)
        self.target_actor = Actor(state_dim, action_dim, params.hidden_dim, max_action).to(params.device)
        self.target_critic = Critic(state_dim, action_dim, params.hidden_dim).to(params.device)

        # Initialize target networks with source network weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params.lr_critic)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.params.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer):
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.params.batch_size)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.params.device)
        actions = torch.FloatTensor(actions).to(self.params.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.params.device)
        next_states = torch.FloatTensor(next_states).to(self.params.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.params.device)

        # --- Update Critic ---
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.params.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft Update Target Networks ---
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.params.tau * param.data + (1.0 - self.params.tau) * target_param.data)
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.params.tau * param.data + (1.0 - self.params.tau) * target_param.data)
            
    def save(self, directory='./models_baseline_1000eps'): # Save to a new directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), os.path.join(directory, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(directory, 'critic.pth'))
        print(f"Model saved to {directory}")

# --- Main Training Function ---
def train():
    params = HyperParams()
    env = MultiRegionEVEnv()
    
    # Get state and action dimensions directly from the environment's spaces
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(state_dim, action_dim, max_action, params)
    replay_buffer = ReplayBuffer(params.replay_capacity)
    
    episode_rewards = []
    
    print(f"--- 开始新一轮训练 (基线参数, 1000 回合) ---")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")

    for episode in tqdm(range(params.max_episodes), desc="训练进度"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            noise = np.random.normal(0, params.noise_std, size=action_dim)
            action = (action + noise).clip(-max_action, max_action)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if replay_buffer.size() > params.batch_size:
                agent.update(replay_buffer)
        
        episode_rewards.append(episode_reward)
        tqdm.write(f"Episode: {episode+1}/{params.max_episodes}, Reward: {episode_reward:.2f}")

    # Save the tuned model and results to new files
    agent.save()
    
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('DDPG Training Rewards (Baseline Params, 1000 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('ddpg_baseline_1000eps_rewards.png')
    plt.show()

if __name__ == '__main__':
    train()
