import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入我们之前创建的环境和智能体结构
from ev_env import MultiRegionEnv
from ddpg_main import Actor # 确保ddpg_main.py在同一目录下

def plot_evaluation_results(results_df, test_day_data, params, day_to_test):
    """
    根据评估结果，为每个区域生成详细的功率平衡图。
    """
    print("\n--- 正在生成详细的评估图表... ---")
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 未找到'SimHei'字体，图例可能显示乱码。")

    for r in params['regions']:
        plt.figure(figsize=(20, 10))
        
        # --- 绘制能源供给部分 (堆叠面积图) ---
        # 准备供给侧的数据
        supply_sources = {
            f'从电网购电': results_df[f'p_buy_{r}'],
            f'光伏发电': results_df[f'pv_used_{r}'],
            f'储能放电': results_df[f'p_discharge_{r}'],
            f'从A区域输入': results_df.get(f'p_trans_A_to_{r}', pd.Series(0, index=results_df.index)),
            f'从B区域输入': results_df.get(f'p_trans_B_to_{r}', pd.Series(0, index=results_df.index)),
            f'从C区域输入': results_df.get(f'p_trans_C_to_{r}', pd.Series(0, index=results_df.index)),
        }
        
        # 移除当前区域自身的输入项
        supply_sources.pop(f'从{r}区域输入', None)
        
        plt.stackplot(results_df['hour'], supply_sources.values(), 
                      labels=supply_sources.keys(), alpha=0.7)

        # --- 绘制能源需求部分 (黑色和红色曲线) ---
        # 总负荷 = 充电负荷 + 居民负荷
        total_static_load = test_day_data[f'charging_load_{r}'] + test_day_data[f'residential_load_{r}']
        
        # 总需求 = 总负荷 + 储能充电 + 向外输电
        total_demand = (total_static_load.values + 
                        results_df[f'p_charge_{r}'] + 
                        results_df[f'p_trans_{r}_to_A'] +
                        results_df[f'p_trans_{r}_to_B'] +
                        results_df[f'p_trans_{r}_to_C'])

        plt.plot(results_df['hour'], total_static_load, label='总负荷 (充电+居民)', color='black', linewidth=2.5)
        plt.plot(results_df['hour'], total_demand, label='总电力需求', color='red', linewidth=2.5, linestyle='--')

        # --- 图表格式化 ---
        plt.title(f'区域 {r} - DDPG智能体在第{day_to_test+1}天的功率平衡调度', fontsize=18)
        plt.xlabel('一天中的小时', fontsize=14)
        plt.ylabel('功率 (kW)', fontsize=14)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(0, 24)
        plt.xticks(range(0, 25, 2))
        plt.tight_layout()
        plt.savefig(f'evaluation_power_dispatch_day_{day_to_test+1}_region_{r}.png')
        plt.close()
        
    print(f"为区域A, B, C生成的详细评估图表已保存。")


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
    
    env = MultiRegionEnv(data_files=DATA_FILES, params=PARAMS)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --- 2. 加载训练好的演员网络 (Actor) ---
    agent_actor = Actor(state_dim, action_dim, max_action)
    try:
        agent_actor.load_state_dict(torch.load("ddpg_model_actor.pth"))
        agent_actor.eval()
        print("成功加载已训练的演员模型 'ddpg_model_actor.pth'。")
    except FileNotFoundError:
        print("错误: 未找到模型文件 'ddpg_model_actor.pth'。请确保它与此脚本在同一目录下。")
        return

    # --- 3. 在指定测试日上运行模拟 ---
    # 我们需要修改环境的reset逻辑，让它能重置到指定的一天
    # 为此，我们直接在外部控制环境的数据
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
    
    # 手动重置环境状态
    env.current_day_data = test_day_data
    env.current_step = 0
    for r in PARAMS['regions']:
        env.current_soc_kwh[r] = PARAMS['ess_initial_soc_val'] * PARAMS['ess_capacity'][r]
    state = env._get_state()
    
    # 存储整个episode的详细结果
    results_history = []
    total_reward = 0

    for t in range(PARAMS['T']):
        action = agent_actor(torch.FloatTensor(state.reshape(1, -1))).cpu().data.numpy().flatten()
        
        # --- 在step之前，手动解析动作以记录详细信息 ---
        step_details = {}
        ess_actions = action[:3]
        trans_actions = action[3:]
        
        for i, r in enumerate(PARAMS['regions']):
            step_details[f'p_charge_{r}'] = max(0, ess_actions[i] * PARAMS['ess_p_charge_max'][r])
            step_details[f'p_discharge_{r}'] = max(0, -ess_actions[i] * PARAMS['ess_p_discharge_max'][r])

        p_trans_map = [('A','B'), ('A','C'), ('B','A'), ('B','C'), ('C','A'), ('C','B')]
        for i, (r_from, r_to) in enumerate(p_trans_map):
            step_details[f'p_trans_{r_from}_to_{r_to}'] = trans_actions[i] * PARAMS['trans_max_power']
        # 补全不存在的传输路径
        for r1 in PARAMS['regions']:
            for r2 in PARAMS['regions']:
                if r1 == r2:
                    step_details[f'p_trans_{r1}_to_{r2}'] = 0

        # 与环境交互
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # 从环境中获取计算出的购售电和弃光信息
        # (这需要稍微修改env.step来返回info字典，为简化，我们在这里重新计算)
        pv_used = {}
        for r in PARAMS['regions']:
            pv_used[r] = (test_day_data.loc[t, f'pv_gen_{r}'] if r in ['A', 'C'] else 0) - env.p_curtail.get(r, 0)
            step_details[f'pv_used_{r}'] = pv_used[r]
            step_details[f'p_buy_{r}'] = env.p_buy.get(r, 0)
            step_details[f'p_sell_{r}'] = env.p_sell.get(r, 0)

        results_history.append(step_details)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
            
    print(f"\n评估完成。测试日总成本: {-total_reward:.2f} 元")

    # --- 4. 可视化评估结果 ---
    results_df = pd.DataFrame(results_history)
    results_df['hour'] = results_df.index * PARAMS['delta_t']
    
    # 调用新的绘图函数
    plot_evaluation_results(results_df, test_day_data, PARAMS, day_to_test)

if __name__ == '__main__':
    # 在运行此脚本前，需要对ev_env.py做一点小修改:
    # 在MultiRegionEnv的step方法的末尾，将p_buy, p_sell, p_curtail存为self属性
    # 例如:
    # self.p_buy = p_buy
    # self.p_sell = p_sell
    # self.p_curtail = p_curtail
    # return next_state, reward, terminated, truncated, info
    
    # 假设ev_env.py已按上述方式修改
    evaluate_agent(day_to_test=300)

