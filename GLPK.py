import subprocess
import sys
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

# --- 步骤 0: 自动安装依赖 ---
# 备注: 此部分会尝试安装必要的Python库。
# GLPK求解器本身需要您单独安装到您的操作系统中。
# 例如，在Windows上，您需要下载GLPK并将其可执行文件路径添加到系统环境变量中。
# 在Linux (Ubuntu/Debian)上，可以运行: sudo apt-get install glpk-utils
# 在macOS (使用Homebrew)上，可以运行: brew install glpk
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

    Args:
        day_index (int): 要模拟的日期列索引 (0代表第一天)。

    Returns:
        pandas.DataFrame or None: 包含所有预处理后输入数据的DataFrame，如果文件未找到则返回None。
    """
    print(f"\n--- 步骤 1: 正在加载第 {day_index+1} 天的数据... ---")
    try:
        # 为没有表头的光伏文件添加 header=None
        pv_a_df = pd.read_csv("One year of solar photovoltaic data A.csv", header=None)
        pv_c_df = pd.read_csv("One year of solar photovoltaic data C.csv", header=None)
        
        # 负荷文件有表头，正常读取
        charging_load_df = pd.read_csv("charging load.csv")
        residential_load_df = pd.read_csv("residential load.csv")
    except FileNotFoundError as e:
        print(f"文件加载错误: {e}。")
        print("错误：请确保所有CSV数据文件 ('One year of solar photovoltaic data A.csv', 'One year of solar photovoltaic data C.csv', 'charging load.csv', 'residential load.csv') 与此脚本位于同一目录下。")
        return None

    # 根据day_index计算负荷数据需要切片的行数范围
    # 一天有96个数据点 (时间步)
    start_row = day_index * 96
    end_row = start_row + 96

    # 检查数据长度是否足够
    if end_row > len(charging_load_df) or end_row > len(residential_load_df):
        print(f"错误: 负荷数据文件行数不足以支持模拟第 {day_index+1} 天。")
        return None

    # 准备一个字典来构建当日数据的DataFrame
    data = {
        'Time_Step': range(96),
        # 充电负荷 (按天切片)
        'charging_load_A': charging_load_df['region A'].iloc[start_row:end_row].values,
        'charging_load_B': charging_load_df['region B'].iloc[start_row:end_row].values,
        'charging_load_C': charging_load_df['region C'].iloc[start_row:end_row].values,
        # 居民负荷 (按天切片)
        'residential_load_A': residential_load_df['A'].iloc[start_row:end_row].values,
        'residential_load_B': residential_load_df['B'].iloc[start_row:end_row].values,
        'residential_load_C': residential_load_df['C'].iloc[start_row:end_row].values,
        # 光伏发电 (使用整数索引 day_index)
        'pv_gen_A': pv_a_df[day_index].values * 100,
        'pv_gen_B': [0] * 96,  # B区域没有光伏
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

    # 时间参数
    params['T'] = 96  # 每日时间步总数 (24 * 4)
    params['delta_t'] = 0.25  # 每个时间步的小时数 (15分钟)

    # 成本和价格参数 (单位: 元/kWh)
    params['cost_curtailment'] = 0.1  # 弃光成本
    params['cost_ess_deg'] = 0.08    # 储能充放电损耗成本
    params['cost_transmission'] = 0.02 # 跨区域单位输电成本

    # 分时电价 (峰、平、谷)
    # 峰: 10:00-12:00, 18:00-21:00
    # 谷: 00:00-07:00
    # 平: 07:00-10:00, 12:00-18:00, 21:00-24:00
    price_buy = {}
    for t in range(96):
        hour = t * 0.25
        if 0 <= hour < 7:
            price_buy[t] = 0.30  # 谷
        elif (10 <= hour < 12) or (18 <= hour < 21):
            price_buy[t] = 0.86  # 峰
        else:
            price_buy[t] = 0.58  # 平
    params['price_buy'] = price_buy
    params['price_sell'] = 0.26 # 统一售电价格

    # 区域和输电参数
    params['regions'] = ['A', 'B', 'C']
    params['trans_loss_factor'] = 0.05 # 5%的传输损耗
    params['trans_max_power'] = 200 # kW, 线路最大传输功率

    # 储能系统 (ESS) 参数
    params['ess_regions'] = ['A', 'B', 'C']
    params['ess_capacity'] = {'A': 450, 'B': 600, 'C': 350}  # kWh
    params['ess_soc_min'] = 0.2
    params['ess_soc_max'] = 0.95
    params['ess_charge_eff'] = 0.95
    params['ess_discharge_eff'] = 0.95
    params['ess_p_charge_max'] = {'A': 150, 'B': 120, 'C': 100} # kW
    params['ess_p_discharge_max'] = {'A': 150, 'B': 120, 'C': 100} # kW
    params['ess_initial_soc_val'] = 0.5 # 初始SOC值

    # 电网互动参数
    params['grid_interact_max'] = {'A': 2000, 'B': 2000, 'C': 2000} # kW

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
    model.T = pyo.Set(initialize=range(params['T']), doc='时间步集合')
    model.R = pyo.Set(initialize=params['regions'], doc='区域集合')
    
    # --- 决策变量 (Variables) ---
    # 与主电网的交互
    model.P_buy = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals, doc='从电网购电功率(kW)')
    model.P_sell = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals, doc='向电网售电功率(kW)')
    # 光伏
    model.P_curtail = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals, doc='弃光功率(kW)')
    # 储能系统
    model.P_charge = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals, doc='储能充电功率(kW)')
    model.P_discharge = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals, doc='储能放电功率(kW)')
    model.SOC = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals, doc='储能荷电状态(kWh)')
    model.u_charge = pyo.Var(model.R, model.T, within=pyo.Binary, doc='储能充电状态(0/1)')
    model.u_discharge = pyo.Var(model.R, model.T, within=pyo.Binary, doc='储能放电状态(0/1)')
    # 区域间传输
    model.P_trans = pyo.Var(model.R, model.R, model.T, within=pyo.NonNegativeReals, doc='区域间传输功率(kW)')

    # --- 目标函数 (Objective Function) ---
    def objective_rule(m):
        cost_buy_grid = sum(params['price_buy'][t] * m.P_buy[r, t] * params['delta_t'] for r in m.R for t in m.T)
        cost_curtail = sum(params['cost_curtailment'] * m.P_curtail[r, t] * params['delta_t'] for r in m.R for t in m.T)
        cost_ess = sum(params['cost_ess_deg'] * (m.P_charge[r, t] + m.P_discharge[r, t]) * params['delta_t'] for r in m.R for t in m.T)
        cost_trans = sum(params['cost_transmission'] * m.P_trans[r_from, r_to, t] * params['delta_t'] for r_from in m.R for r_to in m.R if r_from != r_to for t in m.T)
        revenue_sell_grid = sum(params['price_sell'] * m.P_sell[r, t] * params['delta_t'] for r in m.R for t in m.T)
        return cost_buy_grid + cost_curtail + cost_ess + cost_trans - revenue_sell_grid
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize, doc='最小化总运营成本')

    # --- 约束条件 (Constraints) ---
    # 1. 功率平衡约束
    def power_balance_rule(m, r, t):
        power_in = m.P_buy[r, t] + (data.loc[t, f'pv_gen_{r}'] - m.P_curtail[r,t]) + m.P_discharge[r, t] + \
                   sum(m.P_trans[r_from, r, t] * (1 - params['trans_loss_factor']) for r_from in m.R if r_from != r)
        power_out = m.P_sell[r, t] + m.P_charge[r, t] + data.loc[t, f'charging_load_{r}'] + \
                    data.loc[t, f'residential_load_{r}'] + sum(m.P_trans[r, r_to, t] for r_to in m.R if r_to != r)
        return power_in == power_out
    model.power_balance_con = pyo.Constraint(model.R, model.T, rule=power_balance_rule)

    # 2. 与主电网交互约束
    model.grid_buy_limit_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_buy[r, t] <= params['grid_interact_max'][r])
    model.grid_sell_limit_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_sell[r, t] <= params['grid_interact_max'][r])

    # 3. 光伏发电约束
    model.pv_generation_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_curtail[r, t] <= data.loc[t, f'pv_gen_{r}'])

    # 4. 储能系统(ESS)约束
    def soc_evolution_rule(m, r, t):
        soc_prev = m.SOC[r, t-1] if t > 0 else params['ess_initial_soc_val'] * params['ess_capacity'][r]
        return m.SOC[r, t] == soc_prev + (m.P_charge[r, t] * params['ess_charge_eff'] - m.P_discharge[r, t] / params['ess_discharge_eff']) * params['delta_t']
    model.soc_evolution_con = pyo.Constraint(model.R, model.T, rule=soc_evolution_rule)

    model.soc_min_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.SOC[r, t] >= params['ess_soc_min'] * params['ess_capacity'][r])
    model.soc_max_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.SOC[r, t] <= params['ess_soc_max'] * params['ess_capacity'][r])
    
    model.charge_power_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_charge[r, t] <= params['ess_p_charge_max'][r] * m.u_charge[r, t])
    model.discharge_power_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.P_discharge[r, t] <= params['ess_p_discharge_max'][r] * m.u_discharge[r, t])
    model.charge_discharge_mutex_con = pyo.Constraint(model.R, model.T, rule=lambda m, r, t: m.u_charge[r, t] + m.u_discharge[r, t] <= 1)

    # 5. 区域间电力传输约束
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
    
    # 设置中文字体
    try:
        # 优先使用黑体，如果找不到则使用微软雅黑
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    except:
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("警告: 未找到'SimHei'或'Microsoft YaHei'字体。图例可能显示乱码。")

    # 提取结果到DataFrame
    results_data = {}
    time_steps = list(model.T)
    for r in params['regions']:
        results_data[f'P_buy_{r}'] = [pyo.value(model.P_buy[r, t]) for t in time_steps]
        results_data[f'P_sell_{r}'] = [pyo.value(model.P_sell[r, t]) for t in time_steps]
        results_data[f'P_curtail_{r}'] = [pyo.value(model.P_curtail[r, t]) for t in time_steps]
        results_data[f'P_charge_{r}'] = [pyo.value(model.P_charge[r, t]) for t in time_steps]
        results_data[f'P_discharge_{r}'] = [pyo.value(model.P_discharge[r, t]) for t in time_steps]
        results_data[f'SOC_{r}_%'] = [pyo.value(model.SOC[r, t]) / params['ess_capacity'][r] * 100 for t in time_steps]
        for r_to in params['regions']:
            if r != r_to:
                results_data[f'P_trans_{r}_to_{r_to}'] = [pyo.value(model.P_trans[r, r_to, t]) for t in time_steps]
    
    results_df = pd.DataFrame(results_data, index=[t * params['delta_t'] for t in time_steps])
    results_df.index.name = 'Hour'
    
    # 保存结果到CSV
    results_filename = 'optimal_schedule_results.csv'
    results_df.to_csv(results_filename)
    print(f"详细调度结果已保存到 '{results_filename}'")

    # --- 可视化 ---
    # 1. 各区域功率平衡图
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
        plt.savefig(f'power_dispatch_region_{r}.png')
        plt.close()

    # 2. 储能SOC变化图
    plt.figure(figsize=(18, 8))
    for r in params['regions']:
        plt.plot(results_df.index, results_df[f'SOC_{r}_%'], label=f'区域 {r} SOC', linewidth=2.5)
    plt.title('各区域储能系统SOC变化曲线', fontsize=16)
    plt.xlabel('一天中的小时', fontsize=12)
    plt.ylabel('荷电状态 (SOC %)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 24)
    plt.ylim(0, 100)
    plt.xticks(range(0, 25, 2))
    plt.savefig('ess_soc_profile.png')
    plt.close()

    # 3. 区域间传输功率图
    plt.figure(figsize=(18, 8))
    for r_from in params['regions']:
        for r_to in params['regions']:
            if r_from != r_to:
                plt.plot(results_df.index, results_df[f'P_trans_{r_from}_to_{r_to}'], label=f'从 {r_from} 到 {r_to}', linewidth=2)
    plt.title('区域间电力传输功率', fontsize=16)
    plt.xlabel('一天中的小时', fontsize=12)
    plt.ylabel('传输功率 (kW)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 2))
    plt.savefig('inter_regional_transmission.png')
    plt.close()
    
    # --- 新增图表: 向主电网售电功率图 ---
    plt.figure(figsize=(18, 8))
    for r in params['regions']:
        plt.plot(results_df.index, results_df[f'P_sell_{r}'], label=f'区域 {r} 售电', linewidth=2.5)
    plt.title('各区域向主电网售电功率曲线', fontsize=16)
    plt.xlabel('一天中的小时', fontsize=12)
    plt.ylabel('售电功率 (kW)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 2))
    plt.savefig('sell_to_grid_profile.png')
    plt.close()
    
    print("结果图表已成功保存为PNG文件。")


# --- 步骤 5: 主执行函数 ---
if __name__ == '__main__':
    # 加载数据
    input_data = load_and_prepare_data(day_index=25) # 模拟第一天 (第0列)
    
    if input_data is not None:
        # 获取参数
        model_params = get_parameters()
        # 构建模型
        model = build_model(input_data, model_params)
        
        print("\n--- 步骤 5: 正在使用GLPK求解器求解... ---")
        print("这可能需要几分钟时间，请耐心等待。求解器日志将显示如下：")
        
        try:
            # 创建求解器实例并求解
            solver = pyo.SolverFactory('glpk')
            results = solver.solve(model, tee=True) # tee=True会显示求解器日志
            
            # 检查求解结果
            if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
                print("\n\033[92m求解成功！找到了最优解。\033[0m")
                total_cost = pyo.value(model.objective)
                print(f"最优总运营成本为: {total_cost:.2f} 元")
                
                # 提取并可视化结果
                process_and_visualize_results(model, input_data, model_params)

            elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
                print("\n\033[91m模型不可行！\033[0m")
                print("这意味着在当前参数和数据下，不存在满足所有约束条件的解。")
                print("请检查：1. 负荷是否过高？ 2. 储能或线路容量是否过小？ 3. 参数配置是否合理？")
            else:
                print(f"\n\033[93m求解结束，但未找到最优解。\033[0m")
                print(f"求解器状态: {results.solver.status}")
                print(f"终止条件: {results.solver.termination_condition}")

        except pyo.common.errors.ApplicationError:
            print("\n\033[91m错误: GLPK求解器未找到。\033[0m")
            print("请确保GLPK已经正确安装，并且其可执行文件 (glpsol) 位于系统的PATH环境变量中。")
            print("您可以尝试在终端中直接运行 'glpsol --version' 来检查GLPK是否安装成功。")

