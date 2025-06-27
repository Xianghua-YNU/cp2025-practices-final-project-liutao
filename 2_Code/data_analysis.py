"""
数据后处理与分析模块：用于对模拟结果进行分析。
"""
import numpy as np

# 物理常数 (cgs单位)
G = 6.67430e-8       # 引力常数
c = 2.99792458e10    # 光速
sigma = 5.670374e-5  # Stefan-Boltzmann常数
R_gas = 8.314462e7   # 气体常数
m_H = 1.6735575e-24  # 氢原子质量
M_sun = 1.989e33     # 太阳质量
R_sun = 6.957e10     # 太阳半径
L_sun = 3.828e33     # 太阳光度

# 恒星参数（类太阳恒星）
n = 3.25             # 多方指数
X = 0.70             # 氢质量分数
Y = 0.28             # 氦质量分数
Z = 0.035            # 金属丰度
rho_c = 150          # 中心密度 (g/cm³)
T_c = 1.5e7          # 中心温度 (K)
superadiabatic_threshold = 0.01  # 对流超绝热阈值

def epsilon_pp(rho, T):
    """改进的pp链反应速率"""
    T9 = T * 1e-9
    T9_corr = np.maximum(T9, 1e-6)
    psi = 1 + 1.412e8*(1/X-1)*np.exp(-49.98*T9_corr**(-1/3))
    return 2.38e6 * rho * X**2 * T9_corr**(-2/3) * np.exp(-3.380/T9_corr**(1/3)) * psi

def mean_molecular_weight(T):
    """温度依赖的平均分子量"""
    ionization = 0.5 * (1 + np.tanh((T - 1e6)/2e5))  # 电离分数
    return 1/(2*X + 0.75*Y + 0.5*Z) * (1 - ionization) + 0.5 * ionization

def kappa_es(rho, T):
    """电子散射不透明度"""
    return 0.2 * (1 + X)

def kappa_ff(rho, T):
    """自由-自由吸收不透明度"""
    T_corr = np.maximum(T, 1e3)
    return 1.0e24 * Z * (1 + X) * rho * T_corr**(-3.5)

def kappa_bf(rho, T):
    """束缚-自由吸收不透明度"""
    T_corr = np.maximum(T, 1e3)
    return 4.3e25 * Z * (1 + X) * rho * T_corr**(-3.5)

def total_kappa(rho, T):
    """总不透明度"""
    return kappa_es(rho, T) + kappa_ff(rho, T) + kappa_bf(rho, T)

def calculate_convection_zone(dlogT_dlogP, r, R_star):
    """确定对流区"""
    gamma = 5/3
    nabla_ad = 1 - 1/gamma
    
    convective = (dlogT_dlogP > (nabla_ad + superadiabatic_threshold))
    
    # 确保存在表面对流区
    if np.mean(convective) < 0.05:
        surface_mask = r/R_star > 0.7
        convective[surface_mask] = True
    
    # 边界处理
    convective[0] = convective[1]
    convective[-1] = convective[-2]
    
    return convective, nabla_ad

def calculate_effective_temperature(L_total, R_star):
    """计算有效温度"""
    return (L_total / (4 * np.pi * sigma * R_star**2)) ** 0.25
