"""
主程序入口：用于运行主要的物理模拟。
"""
# main_simulation.py
import numpy as np
from numerical_core import *
from visualization import *
from postprocessing import *
from utils import *

def run_stellar_simulation():
    """运行恒星结构模拟主程序"""
    # 1. 求解Lane-Emden方程
    xi_uniform, theta_uniform, xi_max = solve_lane_emden(n)
    
    # 2. 创建非均匀网格
    r_frac, xi, r, R_star, theta = create_nonuniform_grid(
        xi_uniform, theta_uniform, xi_max, rho_c, T_c
    )
    
    # 3. 计算物理量
    results = compute_physical_quantities(
        r_frac, xi, r, R_star, theta, rho_c, T_c
    )
    rho, P, T, M_r, M_total, K, r, R_star, mu_values = results
    
    # 4. 计算光度
    epsilon, L_r, L_total = calculate_luminosity(r, rho, T)
    
    # 5. 计算不透明度
    kappa = calculate_opacity(rho, T)
    
    # 6. 计算温度梯度
    dlogT_dlogP = calculate_temperature_gradient(T, P)
    
    # 7. 确定对流区
    convective, nabla_ad = determine_convection_zone(dlogT_dlogP, r, R_star)
    
    # 8. 收集参数用于可视化
    params = {
        'n': n,
        'rho_c': rho_c,
        'T_c': T_c,
        'Z': Z,
        'M_total': M_total,
        'R_star': R_star,
        'L_total': L_total
    }
    
    # 9. 可视化
    plot_stellar_structure(
        r, R_star, rho, T, P, M_r, M_total, L_r, L_total, 
        epsilon, convective, dlogT_dlogP, nabla_ad, params
    )
    
    plot_convection_analysis(r, R_star, kappa, dlogT_dlogP, nabla_ad)
    
    # 10. 数据后处理
    print_results_table(r, R_star, rho, T, P, L_r, L_total, kappa, convective)
    print_diagnostics(r, R_star, kappa, T, rho, dlogT_dlogP, nabla_ad, convective, L_total, R_star)

# 运行模拟
if __name__ == "__main__":
    run_stellar_simulation()
