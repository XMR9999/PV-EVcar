# -*- coding: utf-8 -*-
"""
DDPG Agent Testing and Visualization Script.

This script loads a pre-trained DDPG agent and runs it on the environment
for one full episode to analyze its behavior. It then plots the results.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Import the environment and the DDPG agent's network structures
from ev_env import MultiRegionEVEnv
from ddpg_train import Actor, HyperParams

def test_agent(model_dir='./models', start_step=0):
    """
    Loads a trained agent and tests its performance on a specific day.

    Args:
        model_dir (str): Directory where the trained actor.pth is saved.
        start_step (int): The starting step in the dataset for the simulation.
                          Represents the start of the day to test.
    """
    # --- Initialization ---
    env = MultiRegionEVEnv()
    
    # Use a fixed set of hyperparameters consistent with the best model
    params = HyperParams()
    params.hidden_dim = 256 # Match the architecture of the model being loaded
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # We only need the Actor network for testing
    actor = Actor(state_dim, action_dim, params.hidden_dim, max_action).to(params.device)
    
    # --- Load the Trained Model ---
    actor_path = os.path.join(model_dir, 'actor.pth')
    if not os.path.exists(actor_path):
        print(f"错误：在 '{model_dir}' 目录下未找到 'actor.pth' 模型文件。")
        print("请确保您已经成功训练并保存了模型，或者 model_dir 路径正确。")
        return

    print(f"正在从 '{actor_path}' 加载已训练的模型...")
    actor.load_state_dict(torch.load(actor_path, map_location=params.device))
    actor.eval() # Set the actor to evaluation mode

    # --- Run One Episode ---
    # FIX: Reset the environment to the specific start_step
    state, _ = env.reset(start_step=start_step)
    
    done = False
    
    # Lists to store data for plotting
    history = {
        'soc_a': [], 'soc_b': [], 'soc_c': [],
        'grid_buy_a': [], 'grid_buy_b': [], 'grid_buy_c': [],
        'grid_sell_a': [], 'grid_sell_b': [], 'grid_sell_c': [],
        'ess_charge_a': [], 'ess_charge_b': [], 'ess_charge_c': [],
        'ess_discharge_a': [], 'ess_discharge_b': [], 'ess_discharge_c': [],
        'trans_ab': [], 'trans_ac': [], 'trans_bc': [],
        'pv_a': [], 'pv_c': [],
        'load_a': [], 'load_b': [], 'load_c': [],
        'cost': [],
        'reward': []
    }

    total_reward = 0
    
    print(f"\n开始在第 {start_step // 96} 天进行确定性模拟 (无噪声)...")
    while not done:
        # Select action deterministically (no noise)
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(params.device)
        action = actor(state_tensor).cpu().data.numpy().flatten()
        
        # --- Store pre-step data ---
        history['soc_a'].append(env.soc[0]); history['soc_b'].append(env.soc[1]); history['soc_c'].append(env.soc[2])
        
        pv = env.pv_generation[env.current_step]
        ev_load = env.ev_load[env.current_step]
        res_load = env.residential_load[env.current_step]
        history['pv_a'].append(pv[0]); history['pv_c'].append(pv[2])
        history['load_a'].append(ev_load[0] + res_load[0])
        history['load_b'].append(ev_load[1] + res_load[1])
        history['load_c'].append(ev_load[2] + res_load[2])
        
        # Perform action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Decode action for logging (log the action that *led* to this state's reward)
        grid_buy = np.maximum(0, action[:3]) * env.grid_interaction_limit
        grid_sell = np.maximum(0, -action[:3]) * env.grid_interaction_limit
        ess_charge = np.maximum(0, action[3:6]) * env.ess_max_charge_power
        ess_discharge = np.maximum(0, -action[3:6]) * env.ess_max_discharge_power
        trans = action[6:9] * env.transmission_limit
        
        history['grid_buy_a'].append(grid_buy[0]); history['grid_buy_b'].append(grid_buy[1]); history['grid_buy_c'].append(grid_buy[2])
        history['grid_sell_a'].append(grid_sell[0]); history['grid_sell_b'].append(grid_sell[1]); history['grid_sell_c'].append(grid_sell[2])
        history['ess_charge_a'].append(ess_charge[0]); history['ess_charge_b'].append(ess_charge[1]); history['ess_charge_c'].append(ess_charge[2])
        history['ess_discharge_a'].append(ess_discharge[0]); history['ess_discharge_b'].append(ess_discharge[1]); history['ess_discharge_c'].append(ess_discharge[2])
        history['trans_ab'].append(trans[0]); history['trans_ac'].append(trans[1]); history['trans_bc'].append(trans[2])
        
        done = terminated or truncated
        state = next_state
        total_reward += reward
        history['reward'].append(reward)
        history['cost'].append(-reward)

    total_cost = -total_reward
    print(f"\n模拟完成。")
    print(f"总奖励: {total_reward:.2f}")
    print(f"总成本: {total_cost:.2f} 元")

    # --- Plotting the Results ---
    plot_results(history, start_step)

def plot_results(history, start_step):
    """Generates detailed plots of the simulation results."""
    print("正在生成分析图表...")
    
    # --- FIX FOR CHINESE CHARACTERS ---
    try:
        plt.rcParams['font.sans-serif'] = ['DengXian', 'SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"无法设置中文字体，图例可能显示不正确: {e}")

    # The loop runs 96 times, so we should have 96 data points.
    time_steps = np.arange(len(history['soc_a'])) / 4.0
    
    fig, axs = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f'DDPG 智能体行为分析 (第 {start_step // 96} 天)', fontsize=16)

    # 1. SOC Plot
    axs[0].plot(time_steps, history['soc_a'], label='SOC A')
    axs[0].plot(time_steps, history['soc_b'], label='SOC B')
    axs[0].plot(time_steps, history['soc_c'], label='SOC C')
    axs[0].set_ylabel('SOC')
    axs[0].set_ylim(0, 1)
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('储能电池状态 (SOC)')

    # 2. Power Load vs Generation Plot
    axs[1].stackplot(time_steps, history['load_a'], history['load_b'], history['load_c'], 
                     labels=['负荷 A', '负荷 B', '负荷 C'], alpha=0.5)
    axs[1].plot(time_steps, history['pv_a'], label='光伏 A', color='orange', linestyle='--')
    axs[1].plot(time_steps, history['pv_c'], label='光伏 C', color='green', linestyle='--')
    axs[1].set_ylabel('功率 (kW)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title('负荷与光伏发电')

    # 3. ESS Charge/Discharge Plot
    axs[2].plot(time_steps, history['ess_charge_a'], label='充电 A', color='tab:blue')
    axs[2].plot(time_steps, np.array(history['ess_discharge_a']) * -1, label='放电 A', color='tab:blue', linestyle='--')
    axs[2].plot(time_steps, history['ess_charge_b'], label='充电 B', color='tab:orange')
    axs[2].plot(time_steps, np.array(history['ess_discharge_b']) * -1, label='放电 B', color='tab:orange', linestyle='--')
    axs[2].plot(time_steps, history['ess_charge_c'], label='充电 C', color='tab:green')
    axs[2].plot(time_steps, np.array(history['ess_discharge_c']) * -1, label='放电 C', color='tab:green', linestyle='--')
    axs[2].set_ylabel('功率 (kW)')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_title('储能充放电功率 (正=充, 负=放)')

    # 4. Grid Interaction Plot
    total_buy = np.sum([history['grid_buy_a'], history['grid_buy_b'], history['grid_buy_c']], axis=0)
    total_sell = np.sum([history['grid_sell_a'], history['grid_sell_b'], history['grid_sell_c']], axis=0)
    axs[3].plot(time_steps, total_buy, label='总购电')
    axs[3].plot(time_steps, total_sell * -1, label='总售电')
    axs[3].set_ylabel('功率 (kW)')
    axs[3].legend()
    axs[3].grid(True)
    axs[3].set_title('与主电网交互功率 (正=购电, 负=售电)')

    # 5. Transmission Plot
    axs[4].plot(time_steps, history['trans_ab'], label='A -> B')
    axs[4].plot(time_steps, history['trans_ac'], label='A -> C')
    axs[4].plot(time_steps, history['trans_bc'], label='B -> C')
    axs[4].set_xlabel('一天中的小时')
    axs[4].set_ylabel('功率 (kW)')
    axs[4].legend()
    axs[4].grid(True)
    axs[4].set_title('区域间传输功率 (正向流动)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('ddpg_agent_behavior.png')
    plt.show()


if __name__ == '__main__':
    # --- 使用说明 ---
    # 1. model_dir: 指定你希望测试的模型所在的文件夹路径。
    #    - './models' 是我们第一次训练得到的模型。
    #    - './models_baseline_1000eps' 是我们刚刚用奖励塑形训练得到的新模型。
    # 2. start_step: 指定从一年中的哪一刻开始模拟。
    #    我们可以选择一个有代表性的天，比如第180天（夏季）。
    
    # --- FIX: Point to the newly trained model directory ---
    test_agent(model_dir='./models_baseline_1000eps', start_step=96 * 180)
