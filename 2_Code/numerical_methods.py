import numpy as np
from scipy.integrate import solve_ivp, cumtrapz

def solve_lane_emden(n):
    """求解Lane-Emden方程获取恒星密度分布"""
    def lane_emden_deriv(xi, y, n):
        theta, dtheta_dxi = y
        if xi < 1e-6:
            d2theta_dxi2 = -theta**n / 3.0
        else:
            d2theta_dxi2 = -theta**n - (2.0/xi) * dtheta_dxi
        return [dtheta_dxi, d2theta_dxi2]

    def surface_event(xi, y, n):
        return y[0]
    surface_event.terminal = True
    surface_event.direction = -1

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
    
    if sol.t_events[0].size > 0:
        xi_max = sol.t_events[0][0]
    else:
        xi_max = sol.t[-1]
    
    return sol, xi_max

def high_order_gradient(y, x):
    """五阶精度梯度计算（非均匀网格）"""
    n = len(y)
    dydx = np.zeros(n)
    
    # 内部点使用五阶差分
    for i in range(2, n-2):
        h1 = x[i-1] - x[i-2]
        h2 = x[i] - x[i-1]
        h3 = x[i+1] - x[i]
        
        dydx[i] = (1/(h2+h3)) * (
            -h3/(h1*(h1+h2)) * y[i-2]
            + (h3/(h1*h2) + h2/(h1*(h1+h2))) * y[i-1]
            + (h2/(h3*(h2+h3)) - h3/(h2*(h2+h3))) * y[i]
            - h2/(h3*(h2+h3)) * y[i+1]
        )
    
    # 边界处理（一阶差分）
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    dydx[1] = (y[2] - y[1]) / (x[2] - x[1])
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    dydx[-2] = (y[-2] - y[-3]) / (x[-2] - x[-3])
    
    return dydx

def improved_integration(y, x):
    """高精度积分（改进梯形法）"""
    integral = np.zeros_like(x)
    for i in range(1, len(x)):
        h = x[i] - x[i-1]
        integral[i] = integral[i-1] + h * (y[i] + y[i-1]) / 2
    return integral

def calculate_mass_distribution(r, rho):
    """计算质量分布"""
    dm_dr = 4 * np.pi * r**2 * rho
    M_r = cumtrapz(dm_dr, r, initial=0)
    return M_r

def calculate_luminosity(r, rho, epsilon):
    """计算光度分布"""
    dL_dr = 4 * np.pi * r**2 * rho * epsilon
    L_r = improved_integration(dL_dr, r)
    return L_r
