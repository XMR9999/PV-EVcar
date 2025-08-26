import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import sys

# --- 步骤 0: 自动安装依赖 ---
try:
    print("正在检查并安装必要的Python库 (torch, matplotlib)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "matplotlib"])
    print("库已成功安装或已存在。")
except subprocess.CalledProcessError as e:
    print(f"在安装库时发生错误: {e}")
    print("请手动运行 'pip install torch matplotlib' 后再试。")
    sys.exit(1)

from ev_env import MultiRegionEnv
from ddpg_main import Actor

def plot_evaluation_results(results_df, test_day_data, params, day_to_test):
    """
    根据评估结果，为每个区域生成与示例图片一致的功率平衡图。
    """
    print("\n--- 正在生成详细的评估图表... ---")
    
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"警告: 设置中文字体'SimHei'失败: {e}。图例可能显示乱码。")

    colors = {
        'grid_buy': '#377eb8',
        'pv_gen': '#ff7f00',
        'ess_discharge': '#4daf4a',
        'total_load': '#e41a1c'
    }

    for r in params['regions']:
        fig, ax = plt.subplots(figsize=(18, 9))
        
        supply_sources = {
            '从电网购电': results_df[f'p_buy_{r}'],
            '光伏发电': results_df[f'pv_used_{r}'],
            '储能放电': results_df[f'p_discharge_{r}'],
        }
        
        ax.stackplot(results_df['hour'], supply_sources.values(), 
                     labels=supply_sources.keys(),
                     colors=[colors['grid_buy'], colors['pv_gen'], colors['ess_discharge']],
                     alpha=0.8)

        total_load = test_day_data[f'charging_load_{r}'] + test_day_data[f'residential_load_{r}']
        
        ax.plot(results_df['hour'], total_load, 
                label='总负荷 (充电+居民)', 
                color=colors['total_load'], 
                linewidth=2.5, 
                linestyle='--')

        ax.set_title(f'区域 {r} 功率平衡调度图', fontsize=20, pad=20)
        ax.set_xlabel('一天中的小时', fontsize=16)
        ax.set_ylabel('功率 (kW)', fontsize=16)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, 24)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        
        filename = f'power_dispatch_region_{r}.png'
        plt.savefig(filename)
        plt.close(fig)
        print(f"图表已保存为: {filename}")
        
    print(f"\n所有区域的评估图表已生成完毕。")


def evaluate_agent(day_to_test=300):
    """
    加载训练好的DDPG智能体，并在一个指定的测试日上运行，
    最后将调度结果可视化。
    """
    print(f"\n--- 开始评估已训练的DDPG智能体 (测试日: 第{day_to_test+1}天) ---")

    # --- 1. 初始化环境和参数 ---
    DATA_FILES = {
        'pv_a': "One year of solar photovoltaic data A.csv",
        'pv_c': "One year of solar photovoltaic data C.csv",
        'charging_load': "charging load.csv",
        'residential_load': "residential load.csv",
    }
    
    price_buy = {}
    for t in range(96):
        hour = t * 0.25
        if 0 <= hour < 7:
            price_buy[t] = 0.30
        elif (10 <= hour < 12) or (18 <= hour < 21):
            price_buy[t] = 0.86
        else:
            price_buy[t] = 0.58

    PARAMS = {
        'T': 96, 'delta_t': 0.25, 'regions': ['A', 'B', 'C'],
        'price_buy': price_buy,
        'price_sell': 0.26,
        'cost_curtailment': 0.1, 'cost_ess_deg': 0.08, 'cost_transmission': 0.02,
        'trans_loss_factor': 0.05, 'trans_max_power': 200,
        'ess_capacity': {'A': 450, 'B': 600, 'C': 350},
        'ess_soc_min': 0.2, 'ess_soc_max': 0.95,
        'ess_charge_eff': 0.95, 'ess_discharge_eff': 0.95,
        'ess_p_charge_max': {'A': 150, 'B': 120, 'C': 100},
        'ess_p_discharge_max': {'A': 150, 'B': 120, 'C': 100},
        'ess_initial_soc_val': 0.5
    }
    
    env = MultiRegionEnv(data_files=DATA_FILES, params=PARAMS)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --- 2. 加载训练好的演员网络 ---
    agent_actor = Actor(state_dim, action_dim, max_action)
    try:
        agent_actor.load_state_dict(torch.load("ddpg_model_actor.pth"))
        agent_actor.eval()
        print("成功加载已训练的演员模型 'ddpg_model_actor.pth'。")
    except FileNotFoundError:
        print("错误: 未找到模型文件 'ddpg_model_actor.pth'。请确保它与此脚本在同一目录下。")
        return

    # --- 3. 在指定测试日上运行模拟 ---
    start_row = day_to_test * 96
    end_row = start_row + 96
    test_day_data = pd.DataFrame({
        'pv_gen_A': env.pv_a_df[day_to_test].values * 100,
        'pv_gen_C': env.pv_c_df[day_to_test].values * 100,
        'charging_load_A': env.charging_load_df['region A'].iloc[start_row:end_row].values,
        'charging_load_B': env.charging_load_df['region B'].iloc[start_row:end_row].values,
        'charging_load_C': env.charging_load_df['region C'].iloc[start_row:end_row].values,
        'residential_load_A': env.residential_load_df['A'].iloc[start_row:end_row].values,
        'residential_load_B': env.residential_load_df['B'].iloc[start_row:end_row].values,
        'residential_load_C': env.residential_load_df['C'].iloc[start_row:end_row].values,
    })
    
    env.current_day_data = test_day_data
    env.current_step = 0
    for r in PARAMS['regions']:
        env.current_soc_kwh[r] = PARAMS['ess_initial_soc_val'] * PARAMS['ess_capacity'][r]
    state = env._get_state()
    
    results_history = []
    total_reward = 0

    for t in range(PARAMS['T']):
        with torch.no_grad():
            action = agent_actor(torch.FloatTensor(state.reshape(1, -1))).cpu().numpy().flatten()
        
        # ==================== 代码修复处 ====================
        # 裁剪动作以确保其在合法的 [low, high] 范围内
        action = action.clip(env.action_space.low, env.action_space.high)
        # ===================================================

        step_details = {}
        ess_actions = action[:3]
        trans_actions = action[3:]
        
        for i, r in enumerate(PARAMS['regions']):
            step_details[f'p_charge_{r}'] = max(0, ess_actions[i] * PARAMS['ess_p_charge_max'][r])
            step_details[f'p_discharge_{r}'] = max(0, -ess_actions[i] * PARAMS['ess_p_discharge_max'][r])
            step_details[f'soc_{r}_%'] = (env.current_soc_kwh[r] / PARAMS['ess_capacity'][r]) * 100

        step_details['p_trans_A_to_B'] = trans_actions[0] * PARAMS['trans_max_power']
        step_details['p_trans_C_to_B'] = trans_actions[1] * PARAMS['trans_max_power']

        next_state, reward, terminated, truncated, _ = env.step(action)
        
        for r in PARAMS['regions']:
            pv_gen = test_day_data.loc[t, f'pv_gen_{r}'] if r in ['A', 'C'] else 0
            step_details[f'pv_used_{r}'] = pv_gen - env.p_curtail.get(r, 0)
            step_details[f'p_buy_{r}'] = env.p_buy.get(r, 0)
            step_details[f'p_sell_{r}'] = env.p_sell.get(r, 0)
            step_details[f'p_curtail_{r}'] = env.p_curtail.get(r, 0)

        results_history.append(step_details)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
            
    print(f"\n评估完成。测试日总成本: {-total_reward:.2f} 元")

    # --- 4. 整理并保存详细调度结果 ---
    results_df = pd.DataFrame(results_history)
    results_df['hour'] = results_df.index * PARAMS['delta_t']
    
    schedule_df = pd.DataFrame()
    schedule_df['Hour'] = results_df['hour']
    
    for r in PARAMS['regions']:
        schedule_df[f'P_buy_{r}'] = results_df[f'p_buy_{r}']
        schedule_df[f'P_sell_{r}'] = results_df[f'p_sell_{r}']
        schedule_df[f'P_curtail_{r}'] = results_df[f'p_curtail_{r}']
        schedule_df[f'P_charge_{r}'] = results_df[f'p_charge_{r}']
        schedule_df[f'P_discharge_{r}'] = results_df[f'p_discharge_{r}']
        schedule_df[f'SOC_{r}_%'] = results_df[f'soc_{r}_%']

    schedule_df['P_trans_A_to_B'] = results_df['p_trans_A_to_B']
    schedule_df['P_trans_C_to_B'] = results_df['p_trans_C_to_B']
    
    csv_filename = 'optimal_schedule_ddpg.csv'
    schedule_df.to_csv(csv_filename, index=False, float_format='%.6f')
    print(f"详细调度结果已保存到 '{csv_filename}'。")

    # --- 5. 可视化评估结果 ---
    plot_evaluation_results(results_df, test_day_data, PARAMS, day_to_test)


if __name__ == '__main__':
    evaluate_agent(day_to_test=300)
