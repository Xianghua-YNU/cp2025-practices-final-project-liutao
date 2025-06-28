"""
可视化函数模块：用于绘制模拟结果和分析数据。
"""
# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import *

def plot_stellar_structure(r, R_star, rho, T, P, M_r, M_total, L_r, L_total, 
                          epsilon, convective, dlogT_dlogP, nabla_ad, params):
    """绘制恒星结构多参数图"""
    plt.figure(figsize=(18, 14), dpi=100)
    gs = GridSpec(3, 3)
    
    # 1. 密度剖面
    ax1 = plt.subplot(gs[0, 0])
    ax1.semilogy(r/R_star, rho, 'b-', linewidth=2)
    ax1.set_xlabel('相对半径 r/R')
    ax1.set_ylabel('密度 (g/cm³)')
    ax1.set_title('密度剖面')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.axvline(x=0.25, color='r', linestyle='--', alpha=0.5)
    
    # 2. 温度剖面
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(r/R_star, T, 'r-', linewidth=2)
    ax2.set_xlabel('相对半径 r/R')
    ax2.set_ylabel('温度 (K)')
    ax2.set_title('温度剖面')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axvline(x=0.25, color='r', linestyle='--', alpha=0.5)
    
    # 3. 压力剖面
    ax3 = plt.subplot(gs[0, 2])
    ax3.semilogy(r/R_star, P, 'g-', linewidth=2)
    ax3.set_xlabel('相对半径 r/R')
    ax3.set_ylabel('压力 (dyn/cm²)')
    ax3.set_title('压力剖面')
    ax3.grid(True, which='both', linestyle='--', alpha=0.7)
    ax3.axvline(x=0.25, color='r', linestyle='--', alpha=0.5)
    
    # 4. 质量剖面
    ax4 = plt.subplot(gs[1, 0])
    ax4.plot(r/R_star, M_r/M_total, 'm-', linewidth=2)
    ax4.set_xlabel('相对半径 r/R')
    ax4.set_ylabel('相对质量 $M_r/M_{total}$')
    ax4.set_title('累积质量分布')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.axvline(x=0.25, color='r', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    
    # 5. 光度剖面
    ax5 = plt.subplot(gs[1, 1])
    ax5.plot(r/R_star, L_r/L_total, 'c-', linewidth=2)
    ax5.set_xlabel('相对半径 r/R')
    ax5.set_ylabel('相对光度 $L_r/L_{total}$')
    ax5.set_title('光度剖面')
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.axvline(x=0.25, color='r', linestyle='--', alpha=0.5)
    ax5.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    
    # 6. 能量产生率
    ax6 = plt.subplot(gs[1, 2])
    ax6.semilogy(r/R_star, epsilon, 'orange', linewidth=2)
    ax6.set_xlabel('相对半径 r/R')
    ax6.set_ylabel('能量产生率 (erg g⁻¹ s⁻¹)')
    ax6.set_title('核能产生')
    ax6.grid(True, which='both', linestyle='--', alpha=0.7)
    ax6.axvline(x=0.25, color='r', linestyle='--', alpha=0.5)
    
    # 7. 对流区可视化
    ax7 = plt.subplot(gs[2, 0])
    if np.any(convective):
        ax7.fill_between(r/R_star, 0.4, 0.6, where=convective, 
                        color='red', alpha=0.6, label='对流区')
        ax7.fill_between(r/R_star, 0.4, 0.6, where=~convective, 
                        color='blue', alpha=0.2, label='辐射区')
        ax7.legend(loc='best')
    else:
        ax7.text(0.5, 0.5, "无对流区", ha='center', va='center', fontsize=12)
    ax7.set_xlabel('相对半径 r/R')
    ax7.set_title('对流区分布')
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0.3, 0.7)
    ax7.set_yticks([])
    
    # 8. 温度梯度比较
    ax8 = plt.subplot(gs[2, 1])
    ax8.plot(r/R_star, dlogT_dlogP, 'b-', label='实际梯度')
    ax8.axhline(y=nabla_ad, color='r', linestyle='--', label='绝热梯度')
    ax8.axhline(y=nabla_ad + superadiabatic_threshold, color='g', linestyle=':', 
               label='对流阈值')
    ax8.fill_between(r/R_star, nabla_ad, dlogT_dlogP, where=(dlogT_dlogP > nabla_ad),
                    color='red', alpha=0.3, label='对流区域')
    ax8.set_xlabel('相对半径 r/R')
    ax8.set_ylabel('$\\nabla = d\ln T/d\ln P$')
    ax8.set_title('温度梯度比较')
    ax8.legend()
    ax8.grid(True, linestyle='--', alpha=0.7)
    ax8.set_ylim(0, 0.5)
    
    # 9. 物理参数摘要
    ax9 = plt.subplot(gs[2, 2])
    ax9.axis('off')
    
    # 对流诊断
    conv_percent = np.mean(convective) * 100
    core_conv = np.mean(convective[r/R_star < 0.3]) * 100
    surface_conv = np.mean(convective[r/R_star > 0.7]) * 100
    
    # 计算有效温度 (修正)
    T_eff = (L_total / (4 * np.pi * sigma * R_star**2)) ** 0.25
    
    param_text = (
        f"恒星参数摘要:\n\n"
        f"多方指数 n = {params['n']:.2f}\n"
        f"中心密度 ρ_c = {params['rho_c']:.1f} g/cm³\n"
        f"中心温度 T_c = {params['T_c']/1e6:.2f} MK\n"
        f"金属丰度 Z = {params['Z']:.3f}\n"
        f"恒星质量 M = {params['M_total']/M_sun:.3f} M⊙\n"
        f"恒星半径 R = {params['R_star']/R_sun:.3f} R⊙\n"
        f"恒星光度 L = {params['L_total']/L_sun:.3f} L⊙\n"
        f"有效温度 T_eff = {T_eff:.0f} K\n\n"
        f"对流诊断:\n"
        f"  - 对流比例: {conv_percent:.1f}%\n"
        f"  - 核心对流比例: {core_conv:.1f}%\n"
        f"  - 表面对流比例: {surface_conv:.1f}%"
    )
    ax9.text(0.05, 0.5, param_text, fontsize=11, family='monospace')
    
    plt.tight_layout()
    plt.savefig('stellar_structure_improved.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_convection_analysis(r, R_star, kappa, dlogT_dlogP, nabla_ad):
    """绘制对流分析图"""
    plt.figure(figsize=(12, 8))
