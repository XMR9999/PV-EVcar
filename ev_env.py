import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import subprocess
import sys

# --- 步骤 0: 自动安装依赖 ---
try:
    print("正在检查并安装必要的Python库 (gymnasium, pandas)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium", "pandas"])
    print("库已成功安装或已存在。")
except subprocess.CalledProcessError as e:
    print(f"在安装库时发生错误: {e}")
    print("请手动运行 'pip install gymnasium pandas' 后再试。")
    sys.exit(1)


class MultiRegionEnv(gym.Env):
    """
    一个用于多区域能源调度的自定义Gymnasium环境。

    这个环境模拟了三个相互连接的电力区域（A, B, C），每个区域都有
    居民负荷、电动汽车充电负荷，以及可能的本地光伏发电和储能系统。
    一个强化学习智能体（Agent）将通过与此环境交互，学习如何制定最优的
    能源调度策略（储能充放-电、区域间电力传输）以最小化总运营成本。

    - 状态空间 (State Space): 描述了系统在每个时间点的完整情况。
    - 动作空间 (Action Space): 定义了智能体可以执行的连续控制操作。
    - 奖励函数 (Reward Function): 基于每一步操作产生的经济成本来评估动作的好坏。
    """
    
    def __init__(self, data_files, params):
        """
        初始化环境。

        Args:
            data_files (dict): 包含所有数据文件路径的字典。
            params (dict): 包含所有模型超参数的字典。
        """
        super(MultiRegionEnv, self).__init__()

        # --- 1. 加载和处理数据 ---
        self.params = params
        self._load_data(data_files)

        # --- 2. 定义动作空间 (Action Space) ---
        # 智能体需要为9个变量输出连续的控制信号
        # - 3个储能动作: [-1, 1] (负数代表放电, 正数代表充电)
        # - 6个输电动作: [0, 1] (代表输电功率占最大功率的比例)
        action_low = np.array([-1.0] * 3 + [0.0] * 6, dtype=np.float32)
        action_high = np.array([1.0] * 3 + [1.0] * 6, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # --- 3. 定义状态空间 (State Space) ---
        # 状态向量包含13个维度，我们对其进行归一化处理 (值都在0-1之间)
        # 这样有助于神经网络的稳定训练
        self.state_dim = 13
        state_low = np.zeros(self.state_dim, dtype=np.float32)
        state_high = np.ones(self.state_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)

        # --- 4. 初始化环境内部变量 ---
        self.current_step = 0
        self.current_day_data = None
        self.current_soc_kwh = {r: 0.0 for r in self.params['regions']}
        # 为评估脚本初始化额外变量
        self.p_buy = {}
        self.p_sell = {}
        self.p_curtail = {}


    def _load_data(self, data_files):
        """一次性加载所有年度数据到内存中。"""
        try:
            self.pv_a_df = pd.read_csv(data_files['pv_a'], header=None)
            self.pv_c_df = pd.read_csv(data_files['pv_c'], header=None)
            self.charging_load_df = pd.read_csv(data_files['charging_load'])
            self.residential_load_df = pd.read_csv(data_files['residential_load'])
            
            # 计算归一化用的最大值
            self.max_pv = max(self.pv_a_df.max().max(), self.pv_c_df.max().max()) * 100
            self.max_charge_load = self.charging_load_df[['region A', 'region B', 'region C']].max().max()
            self.max_res_load = self.residential_load_df[['A', 'B', 'C']].max().max()
            self.max_buy_price = max(self.params['price_buy'].values())

        except FileNotFoundError as e:
            raise IOError(f"数据文件加载失败: {e}")

    def _get_state(self):
        """获取并归一化当前时间步的状态。"""
        # --- 代码修正处 ---
        # 当到达终止状态时，下一个状态通常是全零或不使用，这里返回全零以避免索引越界
        if self.current_step >= self.params['T']:
            return np.zeros(self.state_dim, dtype=np.float32)

        # 提取当前时间步的数据
        t = self.current_step
        pv_a = self.current_day_data['pv_gen_A'].iloc[t]
        pv_c = self.current_day_data['pv_gen_C'].iloc[t]
        load_a_charge = self.current_day_data['charging_load_A'].iloc[t]
        load_b_charge = self.current_day_data['charging_load_B'].iloc[t]
        load_c_charge = self.current_day_data['charging_load_C'].iloc[t]
        load_a_res = self.current_day_data['residential_load_A'].iloc[t]
        load_b_res = self.current_day_data['residential_load_B'].iloc[t]
        load_c_res = self.current_day_data['residential_load_C'].iloc[t]
        
        # 归一化状态向量
        state = np.array([
            t / (self.params['T'] - 1),  # 时间步
            self.current_soc_kwh['A'] / self.params['ess_capacity']['A'],
            self.current_soc_kwh['B'] / self.params['ess_capacity']['B'],
            self.current_soc_kwh['C'] / self.params['ess_capacity']['C'],
            pv_a / self.max_pv if self.max_pv > 0 else 0,
            pv_c / self.max_pv if self.max_pv > 0 else 0,
            load_a_charge / self.max_charge_load if self.max_charge_load > 0 else 0,
            load_b_charge / self.max_charge_load if self.max_charge_load > 0 else 0,
            load_c_charge / self.max_charge_load if self.max_charge_load > 0 else 0,
            load_a_res / self.max_res_load if self.max_res_load > 0 else 0,
            load_b_res / self.max_res_load if self.max_res_load > 0 else 0,
            load_c_res / self.max_res_load if self.max_res_load > 0 else 0,
            self.params['price_buy'][t] / self.max_buy_price if self.max_buy_price > 0 else 0,
        ], dtype=np.float32)
        
        return state

    def reset(self, seed=None, options=None):
        """重置环境到一个新的随机天，并返回初始状态。"""
        super().reset(seed=seed)
        
        # 随机选择新的一天
        day_index = self.np_random.integers(0, 365)
        
        # 准备当天的数据切片
        start_row = day_index * 96
        end_row = start_row + 96
        self.current_day_data = pd.DataFrame({
            'pv_gen_A': self.pv_a_df[day_index].values * 100,
            'pv_gen_C': self.pv_c_df[day_index].values * 100,
            'charging_load_A': self.charging_load_df['region A'].iloc[start_row:end_row].values,
            'charging_load_B': self.charging_load_df['region B'].iloc[start_row:end_row].values,
            'charging_load_C': self.charging_load_df['region C'].iloc[start_row:end_row].values,
            'residential_load_A': self.residential_load_df['A'].iloc[start_row:end_row].values,
            'residential_load_B': self.residential_load_df['B'].iloc[start_row:end_row].values,
            'residential_load_C': self.residential_load_df['C'].iloc[start_row:end_row].values,
        })
        
        # 重置时间和SOC
        self.current_step = 0
        for r in self.params['regions']:
            self.current_soc_kwh[r] = self.params['ess_initial_soc_val'] * self.params['ess_capacity'][r]
            
        initial_state = self._get_state()
        info = {} # info字典可以用来返回调试信息
        
        return initial_state, info

    def step(self, action):
        """
        执行一个时间步。

        Args:
            action (np.ndarray): 智能体输出的动作向量。

        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
        """
        t = self.current_step
        
        # --- 1. 解码动作 ---
        # 将[-1, 1]或[0, 1]的动作信号转换为实际的功率值 (kW)
        ess_actions = action[:3]
        trans_actions = action[3:]
        
        p_charge = {}
        p_discharge = {}
        for i, r in enumerate(self.params['regions']):
            if ess_actions[i] > 0: # 充电
                p_charge[r] = ess_actions[i] * self.params['ess_p_charge_max'][r]
                p_discharge[r] = 0.0
            else: # 放电
                p_charge[r] = 0.0
                p_discharge[r] = -ess_actions[i] * self.params['ess_p_discharge_max'][r]
        
        p_trans = {
            ('A', 'B'): trans_actions[0] * self.params['trans_max_power'],
            ('A', 'C'): trans_actions[1] * self.params['trans_max_power'],
            ('B', 'A'): trans_actions[2] * self.params['trans_max_power'],
            ('B', 'C'): trans_actions[3] * self.params['trans_max_power'],
            ('C', 'A'): trans_actions[4] * self.params['trans_max_power'],
            ('C', 'B'): trans_actions[5] * self.params['trans_max_power'],
        }

        # --- 2. 计算功率平衡和成本 ---
        total_cost_step = 0
        p_buy = {}
        p_sell = {}
        p_curtail = {}

        for r in self.params['regions']:
            # 计算所有输出功率
            load_total = self.current_day_data[f'charging_load_{r}'].iloc[t] + self.current_day_data[f'residential_load_{r}'].iloc[t]
            trans_out = sum(p_trans.get((r, r_to), 0.0) for r_to in self.params['regions'] if r_to != r)
            power_out = load_total + p_charge[r] + trans_out
            
            # 计算所有输入功率 (除电网和光伏外)
            trans_in = sum(p_trans.get((r_from, r), 0.0) * (1 - self.params['trans_loss_factor']) for r_from in self.params['regions'] if r_from != r)
            
            power_in_controllable = p_discharge[r] + trans_in
            
            # 计算净功率缺口
            net_power_gap = power_out - power_in_controllable
            
            # 平衡光伏和电网
            pv_available = self.current_day_data[f'pv_gen_{r}'].iloc[t] if r in ['A', 'C'] else 0
            
            if net_power_gap > pv_available: # 功率不足，需要购电
                p_buy[r] = net_power_gap - pv_available
                p_sell[r] = 0.0
                p_curtail[r] = 0.0
            else: # 功率富余，可以售电或弃光
                p_buy[r] = 0.0
                surplus = pv_available - net_power_gap
                # 假设所有富余电力都上网
                p_sell[r] = surplus 
                p_curtail[r] = 0.0 # 简化处理，假设电网能全部消纳

            # 累加成本
            total_cost_step += (p_buy[r] * self.params['price_buy'][t] - \
                                p_sell[r] * self.params['price_sell']) * self.params['delta_t']
            total_cost_step += p_curtail[r] * self.params['cost_curtailment'] * self.params['delta_t']
            total_cost_step += (p_charge[r] + p_discharge[r]) * self.params['cost_ess_deg'] * self.params['delta_t']
            total_cost_step += trans_out * self.params['cost_transmission'] * self.params['delta_t']

        # --- 3. 更新状态 ---
        self.current_step += 1

        for r in self.params['regions']:
            soc_prev = self.current_soc_kwh[r]
            soc_next = soc_prev + (p_charge[r] * self.params['ess_charge_eff'] - \
                                   p_discharge[r] / self.params['ess_discharge_eff']) * self.params['delta_t']
            # 确保SOC在物理边界内
            self.current_soc_kwh[r] = np.clip(soc_next, 
                                              self.params['ess_soc_min'] * self.params['ess_capacity'][r],
                                              self.params['ess_soc_max'] * self.params['ess_capacity'][r])
        
        # --- 4. 计算奖励和终止条件 ---
        reward = -total_cost_step
        terminated = self.current_step >= self.params['T']
        truncated = False # 在此环境中，我们不使用truncated
        
        next_state = self._get_state()
        info = {} # 可用于返回调试信息, e.g., {'cost': total_cost_step}
        
        # ==================== 代码修改部分开始 ====================
        # --- 5. 存储额外信息供评估脚本使用 ---
        # 根据evaluate_agent.py的要求，将这些变量存为self属性
        self.p_buy = p_buy
        self.p_sell = p_sell
        self.p_curtail = p_curtail
        # ==================== 代码修改部分结束 ====================

        return next_state, reward, terminated, truncated, info

    def close(self):
        """环境关闭时执行清理工作。"""
        print("环境已关闭。")

if __name__ == '__main__':
    # --- 用于测试环境的示例代码 ---
    print("--- 正在测试自定义环境 MultiRegionEnv ---")
    
    # 1. 定义文件路径和参数 (与您的test.py一致)
    DATA_FILES = {
        'pv_a': "One year of solar photovoltaic data A.csv",
        'pv_c': "One year of solar photovoltaic data C.csv",
        'charging_load': "charging load.csv",
        'residential_load': "residential load.csv",
    }
    
    # 根据图片中的逻辑生成分时电价
    price_buy = {}
    for t in range(96):
        hour = t * 0.25
        if 0 <= hour < 7:
            price_buy[t] = 0.30  # 谷
        elif (10 <= hour < 12) or (18 <= hour < 21):
            price_buy[t] = 0.86  # 峰
        else:
            price_buy[t] = 0.58  # 平

    # 这是一个简化的参数字典，实际使用时应从您的get_parameters()函数获取
    PARAMS = {
        'T': 96, 'delta_t': 0.25, 'regions': ['A', 'B', 'C'],
        'price_buy': price_buy,       # 使用新的分时电价
        'price_sell': 0.26,           # 使用新的售电价格
        'cost_curtailment': 0.1, 'cost_ess_deg': 0.08, 'cost_transmission': 0.02,
        'trans_loss_factor': 0.05, 'trans_max_power': 200,
        'ess_capacity': {'A': 450, 'B': 600, 'C': 350},
        'ess_soc_min': 0.2, 'ess_soc_max': 0.95,
        'ess_charge_eff': 0.95, 'ess_discharge_eff': 0.95,
        'ess_p_charge_max': {'A': 150, 'B': 120, 'C': 100},
        'ess_p_discharge_max': {'A': 150, 'B': 120, 'C': 100},
        'ess_initial_soc_val': 0.5
    }

    # 2. 创建环境实例
    try:
        env = MultiRegionEnv(data_files=DATA_FILES, params=PARAMS)
        print("环境创建成功!")
        print("动作空间:", env.action_space)
        print("状态空间:", env.observation_space)

        # 3. 运行一个完整的episode进行测试
        state, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # 随机选择一个动作进行测试
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            if step_count % 24 == 0: # 每24步打印一次信息
                print(f"Step: {step_count}, Reward: {reward:.2f}")

        print(f"\n测试完成! 总步数: {step_count}, 总奖励: {total_reward:.2f}")
        env.close()

    except Exception as e:
        print(f"\n环境测试失败: {e}")
