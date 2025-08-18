import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 步骤 1 & 2: 参数定义与数据准备 (已修改)
# =============================================================================
print("--- 步骤 1 & 2: 定义模型参数和输入数据 ---")

# 1. 定义集合
# 时间周期为96个15分钟的时间段 (24 * 4 = 96)
T = 96  
regions = ['A', 'B', 'C'] # 区域
lines = {('A', 'B'), ('B', 'C'), ('A', 'C')} # 输电线路

# 2. 定义参数
# 原始的分时电价 (24小时)
hourly_buy_price = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.2, 1.2, 1.2, 1.2, 0.8, 0.8, 
                    0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.8, 0.8, 0.5, 0.5]
# 将每小时电价重复4次，以适应15分钟的时间间隔
buy_price_15min = np.repeat(hourly_buy_price, 4)

params = {
    'T': T,
    'regions': regions,
    'lines': lines,
    # 成本/价格 (元/kWh)
    'buy_price': buy_price_15min, # 使用新的15分钟电价
    'sell_price': 0.3,   # 卖电价格
    'cost_curtailment': 0.1, # 弃光成本
    'cost_ess': 0.05,        # 储能充放电成本
    'cost_transmission': 0.02, # 区域间传输成本
    
    # 储能系统 (ESS) 参数
    'ess_capacity': {'A': 2000, 'B': 1500, 'C': 2000}, # 额定容量 (kWh)
    'ess_p_max': {'A': 500, 'B': 400, 'C': 500},     # 最大充/放电功率 (kW)
    'ess_soc_min': 0.2,  # 最小SOC
    'ess_soc_max': 0.95, # 最大SOC
    'ess_soc_initial': 0.5, # 初始SOC
    'eff_ch': 0.95,      # 充电效率
    'eff_dis': 0.95,     # 放电效率

    # 电网和线路参数
    'grid_p_max': 10000, # 与主电网交互的最大功率 (kW)
    'line_capacity': {('A', 'B'): 800, ('B', 'C'): 800, ('A', 'C'): 800}, # 线路容量 (kW)
    'line_loss_rate': 0.05, # 简化的传输损耗率
}

# 3. 从CSV文件加载所有真实预测数据
# --- 加载电动汽车负荷数据 ---
file_path_ev = r"D:\Desktop\EVcar\PV-EVcar\forecast 1 year load.csv"
try:
    ev_load_df = pd.read_csv(file_path_ev)
    ev_load_df_day1 = ev_load_df.head(T)
    ev_load_df_day1 = ev_load_df_day1.rename(columns={'region A': 'A', 'region B': 'B', 'region C': 'C'})
    P_EV_pred_real = {
        'A': ev_load_df_day1['A'].values,
        'B': ev_load_df_day1['B'].values,
        'C': ev_load_df_day1['C'].values,
    }
    print(f"成功从路径 '{file_path_ev}' 加载真实电动汽车负荷数据。")
except FileNotFoundError:
    print(f"错误: 在路径 '{file_path_ev}' 未找到数据文件。请确保文件路径和名称完全正确。")
    exit()

# --- 加载光伏发电数据 ---
file_path_pv_a = r"D:\Desktop\EVcar\PV-EVcar\One year of solar photovoltaic data A.csv"
file_path_pv_c = r"D:\Desktop\EVcar\PV-EVcar\One year of solar photovoltaic data C.csv"
try:
    # 读取A区光伏数据
    df_pv_a = pd.read_csv(file_path_pv_a, header=None)
    # 修正: 从第0列 (第一列) 提取前96行数据
    pv_a_data = df_pv_a.iloc[:T, 0].values
    print(f"成功从路径 '{file_path_pv_a}' 加载A区光伏数据。")

    # 读取C区光伏数据
    df_pv_c = pd.read_csv(file_path_pv_c, header=None)
    # 修正: 从第0列 (第一列) 提取前96行数据
    pv_c_data = df_pv_c.iloc[:T, 0].values
    print(f"成功从路径 '{file_path_pv_c}' 加载C区光伏数据。")

    # 组合成完整的光伏预测字典
    P_PV_pred_real = {
        'A': pv_a_data,
        'B': np.zeros(T), # B区无光伏
        'C': pv_c_data,
    }
except FileNotFoundError as e:
    print(f"错误: 光伏数据文件未找到。请检查路径。 {e}")
    exit()
except Exception as e:
    print(f"读取光伏数据时发生错误: {e}")
    exit()


# 4. 组合所有数据
np.random.seed(42)
data = {
    'P_L_pred': { # 居民负荷 (kW) - 仍然使用模拟数据
        'A': 500 + 300 * np.sin(np.linspace(0, 2*np.pi, T)),
        'B': 400 + 250 * np.sin(np.linspace(0.5, 2*np.pi+0.5, T)),
        'C': 600 + 400 * np.sin(np.linspace(0.2, 2*np.pi+0.2, T)),
    },
    'P_EV_pred': P_EV_pred_real, # 使用从CSV加载的真实EV数据
    'P_PV_pred': P_PV_pred_real, # 使用从CSV加载的真实PV数据
}
# 确保居民负荷为正
for r in regions:
    data['P_L_pred'][r] = np.maximum(0, data['P_L_pred'][r])

print("数据准备完成。")


# =============================================================================
# 步骤 3: 使用 Pyomo 构建数学模型 (无需修改)
# =============================================================================
print("\n--- 步骤 3: 开始构建Pyomo模型 ---")

model = pyo.ConcreteModel(name="Multi_Region_EV_Scheduling")

# 1. 定义集合
model.T = pyo.RangeSet(0, T - 1)
model.Regions = pyo.Set(initialize=regions)
model.Lines = pyo.Set(initialize=lines, dimen=2)

# 2. 定义决策变量
model.P_buy = pyo.Var(model.Regions, model.T, within=pyo.NonNegativeReals)
model.P_sell = pyo.Var(model.Regions, model.T, within=pyo.NonNegativeReals)
model.P_pv_used = pyo.Var(model.Regions, model.T, within=pyo.NonNegativeReals)
model.P_curtail = pyo.Var(model.Regions, model.T, within=pyo.NonNegativeReals)
model.P_ch = pyo.Var(model.Regions, model.T, within=pyo.NonNegativeReals)
model.P_dis = pyo.Var(model.Regions, model.T, within=pyo.NonNegativeReals)
model.SOC = pyo.Var(model.Regions, model.T, within=pyo.NonNegativeReals)
model.u_ch = pyo.Var(model.Regions, model.T, within=pyo.Binary) 
model.u_dis = pyo.Var(model.Regions, model.T, within=pyo.Binary)
model.P_trans = pyo.Var(model.Lines, model.T, within=pyo.NonNegativeReals)

# 3. 定义目标函数 (最小化总成本)
def total_cost_rule(m):
    cost_buy = sum(params['buy_price'][t] * m.P_buy[r, t] for r in m.Regions for t in m.T)
    cost_curtail = sum(params['cost_curtailment'] * m.P_curtail[r, t] for r in m.Regions for t in m.T)
    cost_ess = sum(params['cost_ess'] * (m.P_ch[r, t] + m.P_dis[r, t]) for r in m.Regions for t in m.T)
    revenue_sell = sum(params['sell_price'] * m.P_sell[r, t] for r in m.Regions for t in m.T)
    cost_trans = sum(params['cost_transmission'] * m.P_trans[i, j, t] for (i,j) in m.Lines for t in m.T)
    return cost_buy + cost_curtail + cost_ess - revenue_sell + cost_trans
model.objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

# 4. 定义约束条件
model.constraints = pyo.ConstraintList()

for t in model.T:
    for r in model.Regions:
        # 功率平衡约束
        power_in = model.P_buy[r, t] + model.P_pv_used[r, t] + model.P_dis[r, t] + \
                   sum(model.P_trans[j, i, t] * (1 - params['line_loss_rate']) for (j, i) in model.Lines if i == r)
        power_out = model.P_sell[r, t] + data['P_L_pred'][r][t] + data['P_EV_pred'][r][t] + model.P_ch[r, t] + \
                    sum(model.P_trans[i, j, t] for (i, j) in model.Lines if i == r)
        model.constraints.add(power_in == power_out)

        # 电网交互约束
        model.constraints.add(model.P_buy[r, t] <= params['grid_p_max'])
        model.constraints.add(model.P_sell[r, t] <= params['grid_p_max'])
        
        # 光伏发电约束
        model.constraints.add(model.P_pv_used[r, t] + model.P_curtail[r, t] == data['P_PV_pred'][r][t])
        
        # 储能约束
        if t == 0:
            soc_prev = params['ess_soc_initial'] * params['ess_capacity'][r]
        else:
            soc_prev = model.SOC[r, t-1]
        dt = 0.25 # 15分钟 = 0.25小时
        model.constraints.add(model.SOC[r, t] == soc_prev + model.P_ch[r, t] * params['eff_ch'] * dt - (model.P_dis[r, t] / params['eff_dis']) * dt)
        
        model.constraints.add(model.SOC[r, t] >= params['ess_soc_min'] * params['ess_capacity'][r])
        model.constraints.add(model.SOC[r, t] <= params['ess_soc_max'] * params['ess_capacity'][r])

        model.constraints.add(model.P_ch[r, t] <= params['ess_p_max'][r] * model.u_ch[r, t])
        model.constraints.add(model.P_dis[r, t] <= params['ess_p_max'][r] * model.u_dis[r, t])

        model.constraints.add(model.u_ch[r, t] + model.u_dis[r, t] <= 1)

for t in model.T:
    for (i, j) in model.Lines:
        model.constraints.add(model.P_trans[i, j, t] <= params['line_capacity'][(i,j)])

print("模型构建完成。")

# =============================================================================
# 步骤 4: 求解模型与结果分析 (无需修改)
# =============================================================================
print("\n--- 步骤 4: 求解模型并分析结果 ---")

# 1. 调用求解器
solver = pyo.SolverFactory('glpk')
results = solver.solve(model, tee=True) 

# 2. 结果分析
if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
    print("\n求解成功，找到最优解！")
    print(f"最小总成本: {model.objective():.2f} 元")

    results_df = {}
    for r in regions:
        df = pd.DataFrame(index=range(T))
        df['Buy_Power(kW)'] = [pyo.value(model.P_buy[r, t]) for t in model.T]
        df['Sell_Power(kW)'] = [pyo.value(model.P_sell[r, t]) for t in model.T]
        df['PV_Used(kW)'] = [pyo.value(model.P_pv_used[r, t]) for t in model.T]
        df['PV_Curtail(kW)'] = [pyo.value(model.P_curtail[r, t]) for t in model.T]
        df['ESS_Charge(kW)'] = [pyo.value(model.P_ch[r, t]) for t in model.T]
        df['ESS_Discharge(kW)'] = [pyo.value(model.P_dis[r, t]) for t in model.T]
        df['SOC(%)'] = [100 * pyo.value(model.SOC[r, t]) / params['ess_capacity'][r] for t in model.T]
        df['Resident_Load(kW)'] = data['P_L_pred'][r]
        df['EV_Load(kW)'] = data['P_EV_pred'][r]
        
        trans_in = [sum(pyo.value(model.P_trans[j, i, t]) for (j,i) in model.Lines if i==r) for t in model.T]
        trans_out = [sum(pyo.value(model.P_trans[i, j, t]) for (i,j) in model.Lines if i==r) for t in model.T]
        df['Trans_Net_In(kW)'] = np.array(trans_in) - np.array(trans_out)
        
        results_df[r] = df
        print(f"\n--- {r}区 调度结果摘要 ---")
        print(results_df[r][['Buy_Power(kW)', 'Sell_Power(kW)', 'ESS_Charge(kW)', 'ESS_Discharge(kW)', 'SOC(%)']].describe())

    # 3. 结果可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False    
    
    fig, axes = plt.subplots(len(regions), 2, figsize=(18, 6 * len(regions)), constrained_layout=True)
    fig.suptitle("多区域电动汽车最优调度结果 (15分钟间隔)", fontsize=20)

    for i, r in enumerate(regions):
        df = results_df[r]
        
        ax1 = axes[i, 0]
        power_sources = ['Buy_Power(kW)', 'PV_Used(kW)', 'ESS_Discharge(kW)']
        power_loads = ['Sell_Power(kW)', 'Resident_Load(kW)', 'EV_Load(kW)', 'ESS_Charge(kW)']
        
        ax1.stackplot(df.index, df[power_sources].T, labels=[s.replace('(kW)','') for s in power_sources], alpha=0.8)
        ax1.plot(df.index, df[power_loads].sum(axis=1), 'r--', label='Total Load', linewidth=2)
        ax1.set_title(f"{r}区 - 功率平衡")
        ax1.set_xlabel("时间步 (15分钟/步)")
        ax1.set_ylabel("功率 (kW)")
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = axes[i, 1]
        ax2.plot(df.index, df['SOC(%)'], 'g-', marker='o', markersize=3, label='SOC')
        ax2.axhline(y=params['ess_soc_min']*100, color='r', linestyle='--', label='Min SOC')
        ax2.axhline(y=params['ess_soc_max']*100, color='r', linestyle='--', label='Max SOC')
        ax2.set_title(f"{r}区 - 储能SOC")
        ax2.set_xlabel("时间步 (15分钟/步)")
        ax2.set_ylabel("SOC (%)")
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True)

    plt.show()

else:
    print("\n求解失败。")
    print("Solver Status:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)
