"""
主程序入口：用于运行主要的物理模拟。
"""
import numpy as np
from core_numerics import solve_lane_emden, high_order_gradient, calculate_mass_distribution, calculate_luminosity
from physics_model import epsilon_pp, total_kappa, calculate_convection_zone, calculate_effective_temperature, M_sun, R_sun, L_sun
from utils import create_nonuniform_grid, interpolate_solution, calculate_physical_quantities, recalculate_scaling
from visualization import plot_stellar_structure, plot_convection_analysis, print_results_table, print_diagnostics

def main():
    # ========================================================
    # 求解Lane-Emden方程
    # ========================================================
    n = 3.25
    sol, xi_max = solve_lane_emden(n)
    
    # 创建均匀网格解
    xi_uniform = np.linspace(1e-6, xi_max, 1000)
    theta_vals = sol.sol(xi_uniform)
    theta_uniform = theta_vals[0]
    
    # ========================================================
    # 创建非均匀物理网格
    # ========================================================
    rho_c = 150
    T_c = 1.5e7
    
    # 初始估计
    K_initial = (R_gas / 0.6) * T_c * rho_c**(-1/n)
    alpha_initial = np.sqrt((n+1)*K_initial/(4*np.pi*G) * rho_c**(1/n - 1))
    R_star_initial = alpha_initial * xi_max
    
    # 创建非均匀网格
    r_frac = create_nonuniform_grid(xi_max)
    
    # 插值解到非均匀网格
    theta = interpolate_solution(xi_uniform, theta_uniform, r_frac, xi_max)
    xi = r_frac * xi_max
    r = r_frac * R_star_initial
    
    # ========================================================
    # 计算物理量
    # ========================================================
    rho, P, T, mu_values = calculate_physical_quantities(rho_c, theta, n, r, T_c)
    
    # ========================================================
    # 计算质量分布
    # ========================================================
    M_r = calculate_mass_distribution(r, rho)
    M_total = M_r[-1]
    
    # ========================================================
    # 重新计算缩放因子
    # ========================================================
    r, alpha, K = recalculate_scaling(rho_c, T, mu_values, n, r_frac, xi_max)
    R_star = r[-1]
    
    # ========================================================
    # 计算光度分布
    # ========================================================
    epsilon = epsilon_pp(rho, T)
    L_r = calculate_luminosity(r, rho, epsilon)
    L_total = L_r[-1]
    
    # 调试输出
    print(f"核心光度: {L_r[0]:.2e} erg/s")
    print(f"中间区域光度: {L_r[len(r)//2]:.2e} erg/s")
    print(f"总光度: {L_total:.2e} erg/s (太阳单位: {L_total/L_sun:.3f} L⊙)")
    
    # ========================================================
    # 计算不透明度
    # ========================================================
    kappa = np.array([total_kappa(rho[i], T[i]) for i in range(len(rho))])
    
    # ========================================================
    # 计算温度梯度
    # ========================================================
    logT = np.log(np.maximum(T, 1e3))
    logP = np.log(np.maximum(P, 1e-10))
    dlogT_dlogP = high_order_gradient(logT, logP) / high_order_gradient(logP, logP)
    dlogT_dlogP = np.nan_to_num(dlogT_dlogP, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ========================================================
    # 确定对流区
    # ========================================================
    convective, nabla_ad = calculate_convection_zone(dlogT_dlogP, r, R_star)
    
    # 计算对流统计
    conv_percent = np.mean(convective) * 100
    core_conv = np.mean(convective[r/R_star < 0.3]) * 100
    surface_conv = np.mean(convective[r/R_star > 0.7]) * 100
    
    # ========================================================
    # 计算有效温度
    # ========================================================
    T_eff = calculate_effective_temperature(L_total, R_star)
    
    # ========================================================
    # 可视化
    # ========================================================
    plot_stellar_structure(
        r, R_star, rho, T, P, M_r, M_total, L_r, L_total, 
        epsilon, convective, dlogT_dlogP, nabla_ad, 
        superadiabatic_threshold, kappa, n, rho_c, T_c, Z, 
        conv_percent, core_conv, surface_conv, T_eff
    )
    
    plot_convection_analysis(
        r, R_star, kappa, dlogT_dlogP, nabla_ad, 
        superadiabatic_threshold, T_eff
    )
    
    # ========================================================
    # 结果输出
    # ========================================================
    print_results_table(r, R_star, rho, T, P, L_r, L_total, kappa, convective)
    print_diagnostics(r, convective, kappa, T, rho, dlogT_dlogP, nabla_ad, surface_conv, T_eff)

if __name__ == "__main__":
    main()
