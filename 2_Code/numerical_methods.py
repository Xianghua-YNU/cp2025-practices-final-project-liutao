# numerical_core.py
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d
from utils import *

def solve_lane_emden(n):
    """求解Lane-Emden方程获取密度分布"""
    def lane_emden_deriv(xi, y, n):
        """Lane-Emden方程的一阶导数系统"""
        theta, dtheta_dxi = y
        
        if xi < 1e-6:
            d2theta_dxi2 = -theta**n / 3.0
        else:
            d2theta_dxi2 = -theta**n - (2.0/xi) * dtheta_dxi
        
        return [dtheta_dxi, d2theta_dxi2]
    
    def surface_event(xi, y, n):
        """表面事件 (θ=0)"""
        theta, _ = y
        return theta
    surface_event.terminal = True
    surface_event.direction = -1

    # 数值求解Lane-Emden方程
    sol = solve_ivp(
        fun=lane_emden_deriv,
        t_span=[1e-6, 20],
        y0=[1.0, 0.0],
        args=(n,),
        events=surface_event,
        method='RK45',
        rtol=1e-6,
        atol=1e-8,
        dense_output=True
    )
    
    # 获取解并确定恒星表面
    if sol.t_events[0].size > 0:
        xi_max = sol.t_events[0][0]
    else:
        xi_max = sol.t[-1]
        
    xi_uniform = np.linspace(1e-6, xi_max, 1000)
    theta_vals = sol.sol(xi_uniform)
    return xi_uniform, theta_vals[0], xi_max

def create_nonuniform_grid(xi_uniform, theta_uniform, xi_max, rho_c, T_c):
    """创建非均匀物理网格 (核心区域加密)"""
    # 1. 计算比例因子α
    K = (R_gas / 0.6) * T_c * rho_c**(-1/n)  # 初始估计
    alpha = np.sqrt((n+1)*K/(4*np.pi*G) * rho_c**(1/n - 1))
    R_star_initial = alpha * xi_max

    # 2. 创建非均匀网格 (核心更密集)
    r_core_frac = np.linspace(0, 0.1, 600)**0.5 * 0.1
    r_mid_frac = np.linspace(0.1, 0.7, 300)
    r_surf_frac = np.linspace(0.7, 1.0, 100)
    
    r_frac = np.concatenate((r_core_frac, r_mid_frac, r_surf_frac))
    r_frac.sort()  # 确保单调递增
    
    # 插值θ到非均匀网格
    theta_interp = interp1d(xi_uniform, theta_uniform, kind='cubic', fill_value="extrapolate")
    theta = theta_interp(r_frac * xi_max)
    
    # 3. 用非均匀网格重新计算物理量
    xi = r_frac * xi_max
    r = r_frac * R_star_initial
    R_star = r[-1]  # 实际恒星半径
    
    return r_frac, xi, r, R_star, theta

def compute_physical_quantities(r_frac, xi, r, R_star, theta, rho_c, T_c):
    """计算物理量分布 (密度、压力、温度等)"""
    # 1. 密度、压力和温度剖面
    rho = rho_c * theta**n
    P = (R_gas / 0.6) * T_c * rho_c**(-1/n) * rho**(1 + 1/n)
    
    # 使用温度依赖的平均分子量
    T = np.zeros_like(r)
    mu_values = np.zeros_like(r)
    for i in range(len(r)):
        mu_values[i] = mean_molecular_weight(T_c * theta[i])  # 从θ估计T
        T[i] = (mu_values[i] / R_gas) * P[i] / rho[i]  # 用状态方程更新T
    
    # 增强的温度迭代 (带收敛检查)
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
    
    # 2. 质量分布
    dm_dr = 4 * np.pi * r**2 * rho
    M_r = cumulative_trapezoid(dm_dr, r, initial=0)
    M_total = M_r[-1]
    
    # 3. 用实际中心参数重新计算K
    K = (R_gas / mu_values[0]) * T[0] * rho_c**(-1/n)
    alpha = np.sqrt((n+1)*K/(4*np.pi*G) * rho_c**(1/n - 1))
    r = r_frac * alpha * xi_max
    R_star = r[-1]  # 最终恒星半径
    
    return rho, P, T, M_r, M_total, K, r, R_star, mu_values

def calculate_luminosity(r, rho, T):
    """计算光度分布"""
    epsilon = epsilon_pp(rho, T)
    dL_dr = 4 * np.pi * r**2 * rho * epsilon
    L_r = improved_integration(dL_dr, r)
    L_total = L_r[-1]
    
    print(f"核心光度: {L_r[0]:.2e} erg/s")
    print(f"中部区域光度: {L_r[len(r)//2]:.2e} erg/s")
    print(f"总光度: {L_total:.2e} erg/s (太阳单位: {L_total/L_sun:.3f} L⊙)")
    
    return epsilon, L_r, L_total

def calculate_opacity(rho, T):
    """计算不透明度分布"""
    kappa = np.zeros_like(rho)
    for i in range(len(rho)):
        kappa[i] = total_kappa(rho[i], T[i])
    return kappa

def calculate_temperature_gradient(T, P):
    """计算温度梯度"""
    logT = np.log(np.maximum(T, 1e3))
    logP = np.log(np.maximum(P, 1e-10))
    dlogT_dlogP = high_order_gradient(logT, logP) / high_order_gradient(logP, logP)
    dlogT_dlogP = np.nan_to_num(dlogT_dlogP, nan=0.0, posinf=0.0, neginf=0.0)
    return dlogT_dlogP

def determine_convection_zone(dlogT_dlogP, r, R_star):
    """确定对流区域"""
    gamma = 5/3
    nabla_ad = 1 - 1/gamma
    
    convective = (dlogT_dlogP > (nabla_ad + superadiabatic_threshold))
    
    # 确保类太阳恒星有表面对流区
    if np.mean(convective) < 0.05:
        surface_mask = r/R_star > 0.7
        convective[surface_mask] = True
    
    convective[0] = convective[1]  # 核心边界
    convective[-1] = convective[-2]  # 表面边界
    
    return convective, nabla_ad
