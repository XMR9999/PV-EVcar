# -*- coding: utf-8 -*-
"""
DDPG Simulation Environment for Multi-Region EV Optimal Scheduling

This script implements the simulation environment based on the user's mathematical model
and provided data files. It follows the OpenAI Gym interface.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class MultiRegionEVEnv(gym.Env):
    """
    Custom Environment for Multi-Region EV Optimal Scheduling.
    This class simulates the power system and calculates costs based on agent's actions.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MultiRegionEVEnv, self).__init__()

        self.N_LOOKAHEAD = 24 # How many future steps to include in the state
        
        self.EXPECTED_STATE_DIM = 2 + 3 + self.N_LOOKAHEAD * (3 + 3 + 3 + 2)

        self._load_parameters()
        self._load_data()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.EXPECTED_STATE_DIM,), dtype=np.float32
        )

        self.current_step = 0
        self.start_step = 0
        self.soc = np.zeros(3)

    def _load_parameters(self):
        """Loads all simulation parameters directly in the code."""
        self.regions = ['A', 'B', 'C']
        self.cost_ess_charge_discharge = 0.08
        self.cost_pv_curtailment = 0.1
        self.cost_transmission = 0.02
        self.price_sell_to_grid = 0.4
        self.ess_capacity = np.array([450.0, 600.0, 350.0])
        self.soc_min = 0.2
        self.soc_max = 0.95
        self.ess_max_charge_power = np.array([150.0, 120.0, 100.0])
        self.ess_max_discharge_power = np.array([150.0, 120.0, 100.0])
        self.ess_charge_eff = 0.95
        self.ess_discharge_eff = 0.95
        self.soc_initial = 0.5
        self.grid_interaction_limit = np.array([2000.0, 2000.0, 2000.0])
        self.transmission_limit = 200.0
        self.transmission_loss_factor = 0.05
        self.time_step_hours = 0.25

    def _load_data(self):
        """Loads and preprocesses all time-series data from absolute paths."""
        try:
            path_ev_load = r"D:\Desktop\EVcar\PV-EVcar\forecast 1 year load.csv"
            path_pv_a = r"D:\Desktop\EVcar\PV-EVcar\One year of solar photovoltaic data A.csv"
            path_pv_c = r"D:\Desktop\EVcar\PV-EVcar\One year of solar photovoltaic data C.csv"
            path_res_load = r"D:\Desktop\EVcar\PV-EVcar\A_B_C_data.csv"

            df_ev = pd.read_csv(path_ev_load, usecols=['region A', 'region B', 'region C']).dropna()
            df_res = pd.read_csv(path_res_load, usecols=['A', 'B', 'C']).dropna()
            pv_a_2d = pd.read_csv(path_pv_a, header=None).values
            pv_c_2d = pd.read_csv(path_pv_c, header=None).values

            pv_a = pv_a_2d.flatten(order='F') * 100
            pv_c = pv_c_2d.flatten(order='F') * 100
            
            min_len = min(len(df_ev), len(df_res), len(pv_a), len(pv_c))
            print(f"所有数据文件将统一截断为最短长度: {min_len}")

            self.ev_load = df_ev.values[:min_len]
            self.residential_load = df_res.values[:min_len]
            
            pv_a = pv_a[:min_len]
            pv_c = pv_c[:min_len]
            pv_b = np.zeros_like(pv_a)
            self.pv_generation = np.vstack([pv_a, pv_b, pv_c]).T

            self.buy_price_hourly = np.array([
                0.06, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 
                0.15, 0.18, 0.20, 0.18, 0.15, 0.14, 0.12, 0.11,
                0.10, 0.12, 0.15, 0.18, 0.20, 0.18, 0.15, 0.12
            ])
            self.buy_price = np.repeat(self.buy_price_hourly, 4)
            self.sell_price = np.full_like(self.buy_price, self.price_sell_to_grid)
            self.total_steps = min_len
            self.simulation_length = 96

        except FileNotFoundError as e:
            print(f"错误：数据文件未找到: {e.filename}。请检查文件路径。")
            exit()
        except Exception as e:
            print(f"加载数据时出错: {e}")
            exit()

    def _get_state(self):
        """Constructs the state vector for the current step."""
        time_features = np.array([
            np.sin(2 * np.pi * (self.current_step % 96) / 96.0),
            np.cos(2 * np.pi * (self.current_step % 96) / 96.0)
        ])
        soc_features = self.soc.copy()

        start = self.current_step
        end = start + self.N_LOOKAHEAD
        
        def get_future_data(data, start_idx, end_idx):
            sliced_data = data[start_idx:end_idx]
            if len(sliced_data) < self.N_LOOKAHEAD:
                padding_size = self.N_LOOKAHEAD - len(sliced_data)
                pad_values = np.tile(data[-1], (padding_size, 1))
                sliced_data = np.vstack([sliced_data, pad_values])
            return sliced_data

        future_pv = get_future_data(self.pv_generation, start, end).flatten()
        future_ev_load = get_future_data(self.ev_load, start, end).flatten()
        future_res_load = get_future_data(self.residential_load, start, end).flatten()
        
        price_start_idx = self.current_step % 96
        future_buy_price = np.tile(self.buy_price, 2)[price_start_idx : price_start_idx + self.N_LOOKAHEAD]
        future_sell_price = np.tile(self.sell_price, 2)[price_start_idx : price_start_idx + self.N_LOOKAHEAD]
        future_prices = np.vstack([future_buy_price, future_sell_price]).T.flatten()
        
        state = np.concatenate([
            time_features, soc_features, future_pv,
            future_ev_load, future_res_load, future_prices
        ])

        if len(state) != self.EXPECTED_STATE_DIM:
            raise ValueError(
                f"State dimension mismatch! Expected {self.EXPECTED_STATE_DIM}, but got {len(state)}"
            )
        return state

    def reset(self, seed=None, start_step=None):
        """
        Resets the environment.
        Can reset to a random start_step (for training) or a specific one (for testing).
        """
        super().reset(seed=seed)
        if start_step is not None:
            self.start_step = start_step
        else:
            self.start_step = np.random.randint(0, self.total_steps - self.simulation_length)
        
        self.current_step = self.start_step
        self.soc = np.full(3, self.soc_initial)
        return self._get_state(), {}

    def step(self, action):
        grid_buy = np.maximum(0, action[:3]) * self.grid_interaction_limit
        grid_sell = np.maximum(0, -action[:3]) * self.grid_interaction_limit
        ess_charge = np.maximum(0, action[3:6]) * self.ess_max_charge_power
        ess_discharge = np.maximum(0, -action[3:6]) * self.ess_max_discharge_power

        trans_ab, trans_ac, trans_bc = action[6:9] * self.transmission_limit
        
        pv = self.pv_generation[self.current_step]
        ev = self.ev_load[self.current_step]
        res = self.residential_load[self.current_step]
        
        power_received_a = max(0, -trans_ab) * (1-self.transmission_loss_factor) + max(0, -trans_ac) * (1-self.transmission_loss_factor)
        power_received_b = max(0, trans_ab) * (1-self.transmission_loss_factor) + max(0, -trans_bc) * (1-self.transmission_loss_factor)
        power_received_c = max(0, trans_ac) * (1-self.transmission_loss_factor) + max(0, trans_bc) * (1-self.transmission_loss_factor)
                           
        power_sent_a = max(0, trans_ab) + max(0, trans_ac)
        power_sent_b = max(0, -trans_ab) + max(0, trans_bc)
        power_sent_c = max(0, -trans_ac) + max(0, -trans_bc)

        supply = np.array([
            grid_buy[0] + pv[0] + ess_discharge[0] + power_received_a,
            grid_buy[1] + pv[1] + ess_discharge[1] + power_received_b,
            grid_buy[2] + pv[2] + ess_discharge[2] + power_received_c,
        ])
        demand = np.array([
            grid_sell[0] + ess_charge[0] + ev[0] + res[0] + power_sent_a,
            grid_sell[1] + ess_charge[1] + ev[1] + res[1] + power_sent_b,
            grid_sell[2] + ess_charge[2] + ev[2] + res[2] + power_sent_c,
        ])
        mismatch = supply - demand
            
        pv_curtailment = np.maximum(0, mismatch)
        power_deficit = np.maximum(0, -mismatch)

        self.soc += (ess_charge * self.ess_charge_eff - ess_discharge / self.ess_discharge_eff) * self.time_step_hours / self.ess_capacity

        # --- REWARD CALCULATION WITH AGGRESSIVE OPPORTUNITY COST ---
        
        # 1. Base cost calculation
        cost_grid_buy = np.sum(grid_buy * self.buy_price[self.current_step % 96] * self.time_step_hours)
        revenue_grid_sell = np.sum(grid_sell * self.sell_price[self.current_step % 96] * self.time_step_hours)
        cost_ess = np.sum(ess_charge + ess_discharge) * self.cost_ess_charge_discharge * self.time_step_hours
        cost_trans = (np.abs(trans_ab) + np.abs(trans_ac) + np.abs(trans_bc)) * self.cost_transmission * self.time_step_hours
        
        base_cost = cost_grid_buy - revenue_grid_sell + cost_ess + cost_trans
        reward = -base_cost

        # 2. Add aggressive opportunity cost penalties
        hour_of_day = (self.current_step % 96) / 4.0
        
        # --- Penalty 1: Massive penalty for PV curtailment (wasted money) ---
        # The penalty is the amount of curtailed energy multiplied by the high sell price
        opportunity_cost_curtailment = np.sum(pv_curtailment) * self.price_sell_to_grid * self.time_step_hours
        reward -= opportunity_cost_curtailment * 2.0 # Make it twice as painful as just losing revenue

        # --- Penalty 2: Penalty for NOT discharging during peak price hours (missed savings) ---
        current_price = self.buy_price[self.current_step % 96]
        if current_price > 0.15: # If it's a peak/high price period
            # Calculate how much energy *could* have been discharged
            potential_discharge = (self.soc - self.soc_min) * self.ess_capacity / self.time_step_hours
            potential_discharge = np.minimum(potential_discharge, self.ess_max_discharge_power)
            
            # Calculate how much discharge was "missed"
            missed_discharge = np.maximum(0, potential_discharge - ess_discharge)
            
            # The penalty is the missed savings
            opportunity_cost_peak = np.sum(missed_discharge * current_price * self.time_step_hours)
            reward -= opportunity_cost_peak

        # 3. Add penalties for constraint violations (Hard constraints)
        penalty = 0
        if np.sum(power_deficit) > 1:
            penalty += np.sum(power_deficit) * 10
        
        soc_violations = np.maximum(0, self.soc_min - self.soc) + np.maximum(0, self.soc - self.soc_max)
        penalty += np.sum(soc_violations) * 500 # Keep this penalty significant but not overwhelming
        self.soc = np.clip(self.soc, self.soc_min, self.soc_max)

        reward -= penalty

        # --- Update state and check if done ---
        self.current_step += 1
        terminated = (self.current_step >= self.start_step + self.simulation_length)
        truncated = False
        next_state = self._get_state()
        
        return next_state, reward, terminated, truncated, {}

if __name__ == '__main__':
    print("正在初始化并测试环境...")
    env = MultiRegionEVEnv()
    state, info = env.reset()
    print("环境初始化完成。")
    print(f"状态维度 (来自环境定义): {env.observation_space.shape[0]}")
    print(f"状态维度 (来自 reset() 的实际样本): {state.shape[0]}")
    print(f"动作维度: {env.action_space.shape}")
    
    done = False
    total_reward = 0
    
    print("\n开始一天的模拟（使用随机动作）...")
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    print("\n模拟完成。")
    print(f"总奖励 (负成本): {total_reward:.2f}")
