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
    (已根据特定约束进行修改)
    """
    
    def __init__(self, data_files, params):
        """
        初始化环境。
        """
        super(MultiRegionEnv, self).__init__()

        self.params = params
        self._load_data(data_files)

        # ==================== 修改点 1: 调整动作空间 ====================
        # 新的动作空间包含:
        # - 3个储能动作 (A, B, C): [-1, 1]
        # - 2个输电动作 (A->B, C->B): [0, 1]
        # 总共5个动作
        action_low = np.array([-1.0] * 3 + [0.0] * 2, dtype=np.float32)
        action_high = np.array([1.0] * 3 + [1.0] * 2, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        # =============================================================

        # 状态空间保持不变 (13个维度)
        self.state_dim = 13
        state_low = np.zeros(self.state_dim, dtype=np.float32)
        state_high = np.ones(self.state_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)

        # 初始化环境内部变量
        self.current_step = 0
        self.current_day_data = None
        self.current_soc_kwh = {r: 0.0 for r in self.params['regions']}
        self.p_buy = {}
        self.p_sell = {}
        self.p_curtail = {}


    def _load_data(self, data_files):
        """一次性加载所有年度数据到内存中，并进行数据清洗。"""
        try:
            self.pv_a_df = pd.read_csv(data_files['pv_a'], header=None).fillna(0)
            self.pv_c_df = pd.read_csv(data_files['pv_c'], header=None).fillna(0)
            self.charging_load_df = pd.read_csv(data_files['charging_load']).fillna(0)
            self.residential_load_df = pd.read_csv(data_files['residential_load']).fillna(0)
            
            self.max_pv = max(self.pv_a_df.max().max(), self.pv_c_df.max().max()) * 100
            self.max_charge_load = self.charging_load_df[['region A', 'region B', 'region C']].max().max()
            self.max_res_load = self.residential_load_df[['A', 'B', 'C']].max().max()
            self.max_buy_price = max(self.params['price_buy'].values())

        except FileNotFoundError as e:
            raise IOError(f"数据文件加载失败: {e}")

    def _get_state(self):
        """获取并归一化当前时间步的状态。"""
        if self.current_step >= self.params['T']:
            return np.zeros(self.state_dim, dtype=np.float32)

        t = self.current_step
        pv_a = self.current_day_data['pv_gen_A'].iloc[t]
        pv_c = self.current_day_data['pv_gen_C'].iloc[t]
        load_a_charge = self.current_day_data['charging_load_A'].iloc[t]
        load_b_charge = self.current_day_data['charging_load_B'].iloc[t]
        load_c_charge = self.current_day_data['charging_load_C'].iloc[t]
        load_a_res = self.current_day_data['residential_load_A'].iloc[t]
        load_b_res = self.current_day_data['residential_load_B'].iloc[t]
        load_c_res = self.current_day_data['residential_load_C'].iloc[t]
        
        state = np.array([
            t / (self.params['T'] - 1),
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
        
        day_index = self.np_random.integers(0, 365)
        
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
        
        self.current_step = 0
        for r in self.params['regions']:
            self.current_soc_kwh[r] = self.params['ess_initial_soc_val'] * self.params['ess_capacity'][r]
            
        initial_state = self._get_state()
        info = {}
        
        return initial_state, info

    def step(self, action):
        """
        执行一个时间步 (已重写以匹配新的约束)。
        """
        t = self.current_step
        
        # ==================== 修改点 2: 重写动作解码和物理逻辑 ====================
        # --- 1. 解码新动作 ---
        ess_actions = action[:3]
        trans_actions = action[3:] # 只有 A->B 和 C->B
        
        p_charge = {}
        p_discharge = {}
        for i, r in enumerate(self.params['regions']):
            if ess_actions[i] > 0:
                p_charge[r] = ess_actions[i] * self.params['ess_p_charge_max'][r]
                p_discharge[r] = 0.0
            else:
                p_charge[r] = 0.0
                p_discharge[r] = -ess_actions[i] * self.params['ess_p_discharge_max'][r]
        
        # 只定义允许的传输路径
        p_trans = {
            ('A', 'B'): trans_actions[0] * self.params['trans_max_power'],
            ('C', 'B'): trans_actions[1] * self.params['trans_max_power'],
        }

        # --- 2. 计算功率平衡和成本 ---
        total_cost_step = 0
        p_buy = {r: 0.0 for r in self.params['regions']}
        p_sell = {r: 0.0 for r in self.params['regions']}
        p_curtail = {r: 0.0 for r in self.params['regions']}

        # --- 区域 A 和 C 的逻辑 (只能向外输电) ---
        for r in ['A', 'C']:
            load_total = self.current_day_data[f'charging_load_{r}'].iloc[t] + self.current_day_data[f'residential_load_{r}'].iloc[t]
            trans_out = p_trans.get((r, 'B'), 0.0) # 只能传输到 B
            power_out = load_total + p_charge[r] + trans_out
            
            power_in_controllable = p_discharge[r] # 没有输入传输
            
            net_power_gap = power_out - power_in_controllable
            pv_available = self.current_day_data[f'pv_gen_{r}'].iloc[t]
            
            if net_power_gap > pv_available:
                p_buy[r] = net_power_gap - pv_available
                p_sell[r] = 0.0
                p_curtail[r] = 0.0
            else:
                p_buy[r] = 0.0
                surplus = pv_available - net_power_gap
                p_sell[r] = surplus 
                p_curtail[r] = 0.0

            total_cost_step += trans_out * self.params['cost_transmission'] * self.params['delta_t']

        # --- 区域 B 的逻辑 (只能接收电力并先存后用) ---
        # 1. 计算总输入功率
        trans_in_b = p_trans.get(('A', 'B'), 0.0) * (1 - self.params['trans_loss_factor']) + \
                     p_trans.get(('C', 'B'), 0.0) * (1 - self.params['trans_loss_factor'])

        # 2. B区域的储能充电只能来自传输的电力
        # 智能体依然决定充电功率，但不能超过传输来的总量
        p_charge['B'] = min(p_charge['B'], trans_in_b)
        
        # 3. B区域的负荷只能由储能放电和电网购电满足
        load_total_b = self.current_day_data[f'charging_load_B'].iloc[t] + self.current_day_data[f'residential_load_B'].iloc[t]
        
        if load_total_b > p_discharge['B']:
            p_buy['B'] = load_total_b - p_discharge['B']
        else:
            # 如果放电多于负荷，多余的电可以卖给电网
            p_sell['B'] = p_discharge['B'] - load_total_b
        
        # --- 统一计算所有区域的成本 ---
        for r in self.params['regions']:
            total_cost_step += (p_buy[r] * self.params['price_buy'][t] - \
                                p_sell[r] * self.params['price_sell']) * self.params['delta_t']
            total_cost_step += p_curtail[r] * self.params['cost_curtailment'] * self.params['delta_t']
            total_cost_step += (p_charge[r] + p_discharge[r]) * self.params['cost_ess_deg'] * self.params['delta_t']
        # ==============================================================================

        # --- 3. 更新状态 ---
        self.current_step += 1

        for r in self.params['regions']:
            soc_prev = self.current_soc_kwh[r]
            soc_next = soc_prev + (p_charge[r] * self.params['ess_charge_eff'] - \
                                   p_discharge[r] / self.params['ess_discharge_eff']) * self.params['delta_t']
            self.current_soc_kwh[r] = np.clip(soc_next, 
                                              self.params['ess_soc_min'] * self.params['ess_capacity'][r],
                                              self.params['ess_soc_max'] * self.params['ess_capacity'][r])
        
        # --- 4. 计算奖励和终止条件 ---
        reward = -total_cost_step
        terminated = self.current_step >= self.params['T']
        truncated = False
        
        next_state = self._get_state()
        info = {}
        
        self.p_buy = p_buy
        self.p_sell = p_sell
        self.p_curtail = p_curtail

        return next_state, reward, terminated, truncated, info

    def close(self):
        """环境关闭时执行清理工作。"""
        print("环境已关闭。")

if __name__ == '__main__':
    # ... (测试代码部分无需修改) ...
    pass
