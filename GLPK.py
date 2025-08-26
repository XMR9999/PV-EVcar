import subprocess
import sys
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
# ==================== 修改点 1: 导入特定的错误类型 ====================
from pyomo.common.errors import ApplicationError
# =====================================================================

# --- 步骤 0: 自动安装依赖 ---
try:
    print("正在检查并安装必要的Python库 (pyomo, matplotlib)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyomo", "matplotlib"])
    print("库已成功安装或已存在。")
except subprocess.CalledProcessError as e:
    print(f"在安装库时发生错误: {e}")
    print("请手动运行 'pip install pyomo matplotlib' 后再试。")
    sys.exit(1)


# --- 步骤 1: 数据加载与预处理 ---
def load_and_prepare_data(day_index=0):
    """
    加载所有CSV数据文件，并为指定的模拟日准备数据。
    """
    print(f"\n--- 步骤 1: 正在加载第 {day_index+1} 天的数据... ---")
    try:
        pv_a_df = pd.read_csv("One year of solar photovoltaic data A.csv", header=None).fillna(0)
        pv_c_df = pd.read_csv("One year of solar photovoltaic data C.csv", header=None).fillna(0)
        charging_load_df = pd.read_csv("charging load.csv").fillna(0)
        residential_load_df = pd.read_csv("residential load.csv").fillna(0)
    except FileNotFoundError as e:
        print(f"文件加载错误: {e}。")
        print("错误：请确保所有CSV数据文件与此脚本位于同一目录下。")
        return None

    start_row = day_index * 96
    end_row = start_row + 96

    if end_row > len(charging_load_df) or end_row > len(residential_load_df):
        print(f"错误: 负荷数据文件行数不足以支持模拟第 {day_index+1} 天。")
        return None

    data = {
        'Time_Step': range(96),
        'charging_load_A': charging_load_df['region A'].iloc[start_row:end_row].values,
        'charging_load_B': charging_load_df['region B'].iloc[start_row:end_row].values,
        'charging_load_C': charging_load_df['region C'].iloc[start_row:end_row].values,
        'residential_load_A': residential_load_df['A'].iloc[start_row:end_row].values,
        'residential_load_B': residential_load_df['B'].iloc[start_row:end_row].values,
        'residential_load_C': residential_load_df['C'].iloc[start_row:end_row].values,
        'pv_gen_A': pv_a_df[day_index].values * 100,
        'pv_gen_B': [0] * 96,
        'pv_gen_C': pv_c_df[day_index].values * 100,
    }
    print("数据加载成功。")
    return pd.DataFrame(data)

# --- 步骤 2: 参数配置 ---
def get_parameters():
    """
    硬编码并返回一个包含所有模型参数的字典。
    """
    print("\n--- 步骤 2: 正在配置模型参数... ---")
    params = {}

    params['T'] = 96
    params['delta_t'] = 0.25
    params['cost_curtailment'] = 0.1
    params['cost_ess_deg'] = 0.08
    params['cost_transmission'] = 0.02

    price_buy = {}
    for t in range(96):
        hour = t * 0.25
        if 0 <= hour < 7:
            price_buy[t] = 0.30
        elif (10 <= hour < 12) or (18 <= hour < 21):
            price_buy[t] = 0.86
        else:
            price_buy[t] = 0.58
    params['price_buy'] = price_buy
    params['price_sell'] = 0.26

    params['regions'] = ['A', 'B', 'C']
    params['trans_loss_factor'] = 0.05
    params['trans_max_power'] = 200

    params['ess_regions'] = ['A', 'B', 'C']
    params['ess_capacity'] = {'A': 450, 'B': 600, 'C': 350}
    params['ess_soc_min'] = 0.2
    params['ess_soc_max'] = 0.95
    params['ess_charge_eff'] = 0.95
    params['ess_discharge_eff'] = 0.95
    params['ess_p_charge_max'] = {'A': 150, 'B': 120, 'C': 100}
    params['ess_p_discharge_max'] = {'A': 150, 'B': 120, 'C': 100}
    params['ess_initial_soc_val'] = 0.5

    params['grid_interact_max'] = {'A': 2000, 'B': 2000, 'C': 2000}

    print("参数配置完成。")
    return params

# --- 步骤 3: 建立优化模型 ---
def build_model(data, params):
    """
    使用Pyomo根据给定的数据和参数构建优化模型。
    """
    print("\n--- 步骤 3: 正在构建Pyomo优化模型... ---")
    model = pyo.ConcreteModel()

    # --- 集合 (Sets) ---
    model.T = pyo.Set(initialize=range(params['T']))
    model.R = pyo.Set(initialize=params['regions'])
    
    # --- 决策变量 (Variables) ---
    model.P_buy = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals)
    model.P_sell = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals)
    model.P_curtail = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals)
    model.P_charge = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals)
    model.P_discharge = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals)
    model.u_charge = pyo.Var(model.R, model.T, within=pyo.Binary)
    model.u_discharge = pyo.Var(model.R, model.T, within=pyo.Binary)
    model.P_trans = pyo.Var(model.R, model.R, model.T, within=pyo.NonNegativeReals)

    # --- 目标函数 (Objective Function) ---
    def objective_rule(m):
        cost_buy_grid = sum(params['price_buy'][t] * m.P_buy[r, t] * params['delta_t'] for r in m.R for t in m.T)
        cost_curtail = sum(params['cost_curtailment'] * m.P_curtail[r, t] * params['delta_t'] for r in m.R for t in m.T)
        cost_ess = sum(params['cost_ess_deg'] * (m.P_charge[r, t] + m.P_discharge[r, t]) * params['delta_t'] for r in m.R for t in m.T)
        cost_trans = sum(params['cost_transmission'] * (m.P_trans['A', 'B', t] + m.P_trans['C', 'B', t]) * params['delta_t'] for t in m.T)
        revenue_sell_grid = sum(params['price_sell'] * m.P_sell[r, t] * params['delta_t'] for r in m.R for t in m.T)
        return cost_buy_grid + cost_curtail + cost_ess + cost_trans - revenue_sell_grid
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # --- 约束条件 (Constraints) ---

    # 输电路径约束
    def transmission_path_rule(m, r_from, r_to, t):
        if (r_from == 'A' and r_to == 'B') or (r_from == 'C' and r_to == 'B'):
            return pyo.Constraint.Skip
        return m.P_trans[r_from, r_to, t] == 0
    model.transmission_path_con = pyo.Constraint(model.R, model.R, model.T, rule=transmission_path_rule)

    # 区域 A 和 C 的功率平衡
    def power_balance_AC_rule(m, r, t):
        if r not in ['A', 'C']:
            return pyo.Constraint.Skip
        
        power_in = m.P_buy[r, t] + (data.loc[t, f'pv_gen_{r}'] - m.P_curtail[r,t]) + m.P_discharge[r, t]
        power_out = m.P_sell[r, t] + m.P_charge[r, t] + data.loc[t, f'charging_load_{r}'] + \
                    data.loc[t, f'residential_load_{r}'] + m.P_trans[r, 'B', t]
        return power_in == power_out
    model.power_balance_AC_con = pyo.Constraint(model.R, model.T, rule=power_balance_AC_rule)

    # 区域 B 的特殊功率平衡
    def load_balance_B_rule(m, t):
        power_supply = m.P_discharge['B', t] + m.P_buy['B', t]
        power_demand = data.loc[t, 'charging_load_B'] + data.loc[t, 'residential_load_B'] + m.P_sell['B', t]
        return power_supply == power_demand
    model.load_balance_B_con = pyo.Constraint(model.T, rule=load_balance_B_rule)

    def charge_balance_B_rule(m, t):
        power_from_transmission = (m.P_trans['A', 'B', t] + m.P_trans['C', 'B', t]) * (1 - params['trans_loss_factor'])
        return m.P_charge['B', t] == power_from_transmission
    model.charge_balance_B_con = pyo.Constraint(model.T, rule=charge_balance_B_rule)

    # 与主电网交互约束
    model.grid_buy_limit_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_buy[r, t] <= params['grid_interact_max'][r])
    model.grid_sell_limit_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_sell[r, t] <= params['grid_interact_max'][r])

    # 光伏发电约束
    model.pv_generation_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_curtail[r, t] <= data.loc[t, f'pv_gen_{r}'])

    # 储能系统(ESS)约束
    def soc_evolution_rule(m, r, t):
        soc_prev = m.SOC[r, t-1] if t > 0 else params['ess_initial_soc_val'] * params['ess_capacity'][r]
        return m.SOC[r, t] == soc_prev + (m.P_charge[r, t] * params['ess_charge_eff'] - m.P_discharge[r, t] / params['ess_discharge_eff']) * params['delta_t']
    model.soc_evolution_con = pyo.Constraint(model.R, model.T, rule=soc_evolution_rule)

    model.soc_min_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.SOC[r, t] >= params['ess_soc_min'] * params['ess_capacity'][r])
    model.soc_max_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.SOC[r, t] <= params['ess_soc_max'] * params['ess_capacity'][r])
    
    model.charge_power_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_charge[r, t] <= params['ess_p_charge_max'][r] * m.u_charge[r, t])
    model.discharge_power_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_discharge[r, t] <= params['ess_p_discharge_max'][r] * m.u_discharge[r, t])
    model.charge_discharge_mutex_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.u_charge[r, t] + m.u_discharge[r, t] <= 1)

    # 区域间电力传输约束
    model.transmission_limit_con = pyo.Constraint(model.R, model.R, model.T, rule=lambda m, r_from, r_to, t: m.P_trans[r_from, r_to, t] <= params['trans_max_power'])
    model.self_trans_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_trans[r, r, t] == 0)

    print("模型构建完成。")
    return model

# --- 步骤 4: 结果提取与可视化 ---
def process_and_visualize_results(model, data, params):
    """
    从求解后的模型中提取结果，保存到CSV并生成图表。
    """
    print("\n--- 步骤 4: 正在提取并可视化结果... ---")
    
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 未找到'SimHei'字体。图例可能显示乱码。")

    results_data = {}
    time_steps = list(model.T)
    for r in params['regions']:
        results_data[f'P_buy_{r}'] = [pyo.value(model.P_buy[r, t]) for t in time_steps]
        results_data[f'P_sell_{r}'] = [pyo.value(model.P_sell[r, t]) for t in time_steps]
        results_data[f'P_curtail_{r}'] = [pyo.value(model.P_curtail[r, t]) for t in time_steps]
        results_data[f'P_charge_{r}'] = [pyo.value(model.P_charge[r, t]) for t in time_steps]
        results_data[f'P_discharge_{r}'] = [pyo.value(model.P_discharge[r, t]) for t in time_steps]
        results_data[f'SOC_{r}_%'] = [pyo.value(model.SOC[r, t]) / params['ess_capacity'][r] * 100 for t in time_steps]
    
    results_data['P_trans_A_to_B'] = [pyo.value(model.P_trans['A', 'B', t]) for t in time_steps]
    results_data['P_trans_C_to_B'] = [pyo.value(model.P_trans['C', 'B', t]) for t in time_steps]

    results_df = pd.DataFrame(results_data, index=[t * params['delta_t'] for t in time_steps])
    results_df.index.name = 'Hour'
    
    results_filename = 'optimal_schedule_glpk_constrained.csv'
    results_df.to_csv(results_filename)
    print(f"详细调度结果已保存到 '{results_filename}'")

    # 可视化
    for r in params['regions']:
        plt.figure(figsize=(18, 8))
        sources = {
            '从电网购电': results_df[f'P_buy_{r}'],
            '光伏发电': data[f'pv_gen_{r}'].values - results_df[f'P_curtail_{r}'],
            '储能放电': results_df[f'P_discharge_{r}']
        }
        plt.stackplot(results_df.index, sources.values(), labels=sources.keys(), alpha=0.8)
        total_load = data[f'charging_load_{r}'] + data[f'residential_load_{r}']
        plt.plot(results_df.index, total_load, label='总负荷 (充电+居民)', color='red', linestyle='--', linewidth=2.5)
        plt.title(f'区域 {r} 功率平衡调度图', fontsize=16)
        plt.xlabel('一天中的小时', fontsize=12)
        plt.ylabel('功率 (kW)', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(0, 24)
        plt.xticks(range(0, 25, 2))
        plt.savefig(f'glpk_region_{r}_constrained.png')
        plt.close()

    print("结果图表已成功保存为PNG文件。")


# --- 步骤 5: 主执行函数 ---
if __name__ == '__main__':
    input_data = load_and_prepare_data(day_index=0)
    
    if input_data is not None:
        model_params = get_parameters()
        model = build_model(input_data, model_params)
        
        print("\n--- 步骤 5: 正在使用GLPK求解器求解... ---")
        
        try:
            solver = pyo.SolverFactory('glpk')
            
            # ==================== 修改点 2: 使用正确的超时命令 ====================
            # GLPK v4.65 使用 'tmlim' 而不是 'timelimit'
            solver.options['tmlim'] = 120  # 设置120秒超时
            # =================================================================

            results = solver.solve(model, tee=True)
            
            if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
                print("\n\033[92m求解成功！找到了一个高质量的可行解。\033[0m")
                total_cost = pyo.value(model.objective)
                print(f"最优总运营成本为: {total_cost:.2f} 元")
                process_and_visualize_results(model, input_data, model_params)
            elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
                print("\n\0391m模型不可行！\033[0m")
            else:
                print(f"\n\033[93m求解结束，但未找到最优解。\033[0m")
                print(f"求解器状态: {results.solver.status}")
                print(f"终止条件: {results.solver.termination_condition}")
        # ==================== 修改点 3: 使用正确的错误处理 ====================
        except ApplicationError:
        # =====================================================================
            print("\n\033[91m错误: GLPK求解器未找到或执行失败。\033[0m")
            print("请确保GLPK已经正确安装，并且其可执行文件 (glpsol) 位于系统的PATH环境变量中。")

