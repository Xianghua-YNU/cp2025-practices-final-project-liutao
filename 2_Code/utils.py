"""
工具函数模块：包含一些通用的辅助函数。
"""

import numpy as np
from scipy.interpolate import interp1d

def create_nonuniform_grid(xi_max):
    """创建非均匀物理网格（核心加密）"""
    # 核心区域 (0-10%半径): 60%的点
    r_core_frac = np.linspace(0, 0.1, 600)**0.5 * 0.1
    # 中间区域 (10-70%): 30%的点
    r_mid_frac = np.linspace(0.1, 0.7, 300)
    # 表面区域 (70-100%): 10%的点
    r_surf_frac = np.linspace(0.7, 1.0, 100)
    
    # 合并区域
    r_frac = np.concatenate((r_core_frac, r_mid_frac, r_surf_frac))
    r_frac.sort()
    
    return r_frac

def interpolate_solution(xi_uniform, theta_uniform, r_frac, xi_max):
    """插值解到非均匀网格"""
    theta_interp = interp1d(xi_uniform, theta_uniform, kind='cubic', fill_value="extrapolate")
    return theta_interp(r_frac * xi_max)

def calculate_physical_quantities(rho_c, theta, n, r, T_c):
    """计算物理量（密度、压力、温度）"""
    rho = rho_c * theta**n
    P = (R_gas / 0.6) * T_c * rho_c**(-1/n) * rho**(1 + 1/n)
    
    # 初始化温度和分子量
    T = np.zeros_like(r)
    mu_values = np.zeros_like(r)
    for i in range(len(r)):
        mu_values[i] = mean_molecular_weight(T_c * theta[i])
        T[i] = (mu_values[i] / R_gas) * P[i] / rho[i]
    
    # 温度迭代
    max_iter = 15
    tol = 1e-7
    for iter in range(max_iter):
        T_old = T.copy()
        for i in range(len(r)):
            mu_values[i] = mean_molecular_weight(T[i])
            T[i] = (mu_values[i] / R_gas) * P[i] / rho[i]
        
        delta = np.max(np.abs((T - T_old) / (T_old + 1e-10)))
        if delta < tol:
            print(f"温度计算在迭代 {iter+1} 收敛, 最大变化: {delta:.2e}")
            break
        elif iter == max_iter - 1:
            print(f"温度计算在 {max_iter} 次迭代后未收敛, 最大变化: {delta:.2e}")
    
    return rho, P, T, mu_values

def recalculate_scaling(rho_c, T, mu_values, n, r_frac, xi_max):
    """重新计算缩放因子"""
    K = (R_gas / mu_values[0]) * T[0] * rho_c**(-1/n)
    alpha = np.sqrt((n+1)*K/(4*np.pi*G) * rho_c**(1/n - 1))
    r = r_frac * alpha * xi_max
    return r, alpha, K
