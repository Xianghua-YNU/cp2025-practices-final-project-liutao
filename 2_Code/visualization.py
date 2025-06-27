"""
可视化函数模块：用于绘制模拟结果和分析数据。
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def plot_stellar_structure(r, R_star, rho, T, P, M_r, M_total, L_r, L_total, 
                          epsilon, convective, dlogT_dlogP, nabla_ad, 
                          superadiabatic_threshold, kappa, n, rho_c, T_c, Z, 
                          conv_percent, core_conv, surface_conv, T_eff):
    """绘制恒星结构图"""
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
    
    # 4. 质量分布
    ax4 = plt.subplot(gs[1, 0])
    ax4.plot(r/R_star, M_r/M_total, 'm-', linewidth=2)
    ax4.set_xlabel('相对半径 r/R')
    ax4.set_ylabel('相对质量 $M_r/M_{total}$')
    ax4.set_title('累积质量分布')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.axvline(x=0.25, color='r', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    
    # 5. 光度分布
    ax5 = plt.subplot(gs[1, 1])
    ax5.plot(r/R_star, L_r/L_total, 'c-', linewidth=2)
    ax5.set_xlabel('相对半径 r/R')
    ax5.set_ylabel('相对光度 $L_r/L_{total}$')
    ax5.set_title('光度分布')
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
                    color='red', alpha=0.3, label='对流区')
    ax8.set_xlabel('相对半径 r/R')
    ax8.set_ylabel('$\\nabla = d\ln T/d\ln P$')
    ax8.set_title('温度梯度比较')
    ax8.legend()
    ax8.grid(True, linestyle='--', alpha=0.7)
    ax8.set_ylim(0, 0.5)
    
    # 9. 物理参数摘要
    ax9 = plt.subplot(gs[2, 2])
    ax9.axis('off')
    
    param_text = (
        f"恒星参数摘要:\n\n"
        f"多方指数 n = {n:.2f}\n"
        f"中心密度 ρ_c = {rho_c:.1f} g/cm³\n"
        f"中心温度 T_c = {T_c/1e6:.2f} MK\n"
        f"金属丰度 Z = {Z:.3f}\n"
        f"恒星质量 M = {M_total/M_sun:.3f} M⊙\n"
        f"恒星半径 R = {R_star/R_sun:.3f} R⊙\n"
        f"恒星光度 L = {L_total/L_sun:.3f} L⊙\n"
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

def plot_convection_analysis(r, R_star, kappa, dlogT_dlogP, nabla_ad, superadiabatic_threshold, T_eff):
    """绘制对流分析图"""
    plt.figure(figsize=(12, 8))
    
    # 不透明度分布
    plt.subplot(211)
    plt.semilogy(r/R_star, kappa, 'r-')
    plt.ylabel('不透明度 (cm²/g)')
    plt.title('不透明度分布')
    plt.grid(True)
    
    # 对流稳定性分析
    plt.subplot(212)
    plt.plot(r/R_star, dlogT_dlogP, 'b-', label='∇')
    plt.axhline(nabla_ad, color='r', linestyle='--', label='∇_ad')
    plt.axhline(nabla_ad + superadiabatic_threshold, color='g', linestyle=':', 
               label='对流阈值')
    plt.fill_between(r/R_star, nabla_ad, dlogT_dlogP, 
                     where=(dlogT_dlogP > nabla_ad), 
                     color='red', alpha=0.3)
    plt.ylabel('dlnT/dlnP')
    plt.xlabel('相对半径 r/R')
    plt.legend()
    plt.title('对流稳定性分析')
    plt.grid(True)
    plt.savefig('convection_analysis.png', dpi=300)
    plt.show()

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
        
        region = "对流区" if convective[i] else "辐射区"
        
        print(f"{r_frac:.4f}{rho_val:>12.3f}{T_val:>10.3f}{P_val:>10.2e}{L_frac:>8.3f}{kappa_val:>10.1f}{region:>12}")
    
    print("=" * 85)
    print(f"注: 核心区域 (r/R < {r[indices[1]]/R_star:.2f}) 产生总光度的 {L_r[indices[1]]/L_total:.1%}")

def print_diagnostics(r, convective, kappa, T, rho, dlogT_dlogP, nabla_ad, surface_conv, T_eff):
    """打印诊断信息"""
    conv_percent = np.mean(convective) * 100
    print("\n对流区分析:")
    print("=" * 50)
    print(f"最大不透明度: {np.max(kappa):.1f} cm²/g")
    print(f"表面温度: {T[-1]:.0f} K")
    print(f"表面密度: {rho[-1]:.2e} g/cm³")
    print(f"最大超绝热值: {np.max(dlogT_dlogP - nabla_ad):.4f}")
    print(f"对流点数: {np.sum(convective)}/{len(r)} ({conv_percent:.1f}%)")
    print(f"表面对流比例: {surface_conv:.1f}%")
    print(f"有效温度: {T_eff:.0f} K")
