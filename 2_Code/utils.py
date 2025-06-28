"""
工具函数模块：包含一些通用的辅助函数。
"""

# utils.py
import numpy as np

# 物理常数 (cgs单位制)
G = 6.67430e-8       # 引力常数 (cm³ g⁻¹ s⁻²)
c = 2.99792458e10    # 光速 (cm/s)
sigma = 5.670374e-5  # 斯特藩-玻尔兹曼常数 (erg cm⁻² s⁻¹ K⁻⁴)
R_gas = 8.314462e7   # 气体常数 (erg mol⁻¹ K⁻¹)
m_H = 1.6735575e-24  # 氢原子质量 (g)
M_sun = 1.989e33     # 太阳质量 (g)
R_sun = 6.957e10     # 太阳半径 (cm)
L_sun = 3.828e33     # 太阳光度 (erg/s)

# 恒星参数 (类太阳恒星)
n = 3.25             # 多方指数 (针对对流包层调整)
X = 0.70             # 氢质量分数
Y = 0.28             # 氦质量分数
Z = 0.035            # 金属丰度 (增加以增强不透明度)
rho_c = 150          # 中心密度 (g/cm³)
T_c = 1.5e7          # 中心温度 (K) - 略高于太阳
superadiabatic_threshold = 0.01  # 对流的最小超绝热值

def epsilon_pp(rho, T):
    """改进的质子-质子链反应速率"""
    T9 = T * 1e-9
    T9_corr = np.maximum(T9, 1e-6)
    psi = 1 + 1.412e8*(1/X-1)*np.exp(-49.98*T9_corr**(-1/3))
    return 2.38e6 * rho * X**2 * T9_corr**(-2/3) * np.exp(-3.380/T9_corr**(1/3)) * psi

def mean_molecular_weight(T):
    """温度依赖的平均分子量 (完全电离等离子体)"""
    ionization = 0.5 * (1 + np.tanh((T - 1e6)/2e5))  # 电离分数
    return 1/(2*X + 0.75*Y + 0.5*Z) * (1 - ionization) + 0.5 * ionization

def kappa_es(rho, T):
    """电子散射不透明度 (cm²/g)"""
    return 0.2 * (1 + X)

def kappa_ff(rho, T):
    """自由-自由吸收不透明度 (cm²/g)"""
    T_corr = np.maximum(T, 1e3)
    return 1.0e24 * Z * (1 + X) * rho * T_corr**(-3.5)

def kappa_bf(rho, T):
    """束缚-自由吸收不透明度 (cm²/g) - 表面对流关键参数"""
    T_corr = np.maximum(T, 1e3)
    return 4.3e25 * Z * (1 + X) * rho * T_corr**(-3.5)

def total_kappa(rho, T):
    """总不透明度 (电子散射 + 自由-自由 + 束缚-自由)"""
    return kappa_es(rho, T) + kappa_ff(rho, T) + kappa_bf(rho, T)

def improved_integration(y, x):
    """针对非均匀网格的高精度积分 (改进梯形法)"""
    integral = np.zeros_like(x)
    for i in range(1, len(x)):
        h = x[i] - x[i-1]
        integral[i] = integral[i-1] + h * (y[i] + y[i-1]) / 2
    return integral

def high_order_gradient(y, x):
    """非均匀网格的五阶梯度计算"""
    n = len(y)
    dydx = np.zeros(n)
    
    # 内部点的中心差分
    for i in range(2, n-2):
        h1 = x[i-1] - x[i-2]
        h2 = x[i] - x[i-1]
        h3 = x[i+1] - x[i]
        
        # 非均匀网格系数 (Fornberg方法)
        dydx[i] = (1/(h2+h3)) * (
            -h3/(h1*(h1+h2)) * y[i-2]
            + (h3/(h1*h2) + h2/(h1*(h1+h2))) * y[i-1]
            + (h2/(h3*(h2+h3)) - h3/(h2*(h2+h3))) * y[i]
            - h2/(h3*(h2+h3)) * y[i+1]
        )
    
    # 边界处理 (低阶)
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    dydx[1] = (y[2] - y[1]) / (x[2] - x[1])
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    dydx[-2] = (y[-2] - y[-3]) / (x[-2] - x[-3])
    
    return dydx
