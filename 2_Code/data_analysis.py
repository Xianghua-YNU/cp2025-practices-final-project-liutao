"""
数据后处理与分析模块：用于对模拟结果进行分析。
"""
# postprocessing.py
import numpy as np
from utils import *

def print_results_table(r, R_star, rho, T, P, L_r, L_total, kappa, convective):
    """打印结果表格"""
    indices = [0, len(r)//10, len(r)//4, len(r)//2, 3*len(r)//4, -2]
    
    print("\n恒星内部结构参数:")
    print("=" * 85)
    print(f"{'r/R':<8}{'ρ(g/cm³)':<12}{'T(MK)':<10}{'P(10^17)':<10}{'L_r/L':<8}{'Opacity':<10}{'Zone':<12}")
    print("-" * 85)
    
    for i in indices:
        r_frac = r[i] / R_star
        rho_val = rho[i]
        T_val = T[i] / 1e6
        P_val = P[i] / 1e17
        L_frac = L_r[i] / L_total if L_total > 0 else 0
        kappa_val = kappa[i]
        
        region = "对流" if convective[i] else "辐射"
        
        print(f"{r_frac:.4f}{rho_val:>12.3f}{T_val:>10.3f}{P_val:>10.2e}{L_frac:>8.3f}{kappa_val:>10.1f}{region:>12}")
    
    print("=" * 85)
    print(f"注: 核心区域 (r/R < {r[indices[1]]/R_star:.2f}) 产生了 {L_r[indices[1]]/L_total:.1%} 的总光度")

def print_diagnostics(r, R_star, kappa, T, rho, dlogT_dlogP, nabla_ad, convective, L_total, R_star):
    """打印诊断信息"""
    conv_percent = np.mean(convective) * 100
    surface_conv = np.mean(convective[r/R_star > 0.7]) * 100
    T_eff = (L_total / (4 * np.pi * sigma * R_star**2)) ** 0.25
    
    print("\n对流区分析:")
    print("=" * 50)
    print(f"最大不透明度: {np.max(kappa):.1f} cm²/g")
    print(f"表面温度: {T[-1]:.0f} K")
    print(f"表面密度: {rho[-1]:.2e} g/cm³")
    print(f"最大超绝热性: {np.max(dlogT_dlogP - nabla_ad):.4f}")
    print(f"对流点数: {np.sum(convective)}/{len(r)} ({conv_percent:.1f}%)")
    print(f"表面对流比例: {surface_conv:.1f}%")
    print(f"有效温度: {T_eff:.0f} K")
