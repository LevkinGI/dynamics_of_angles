# simulator.py
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from constants import gamma, h_IFE, delta_t
from config import SimParams

@njit(fastmath=True)
def dynamics(t, y, H, kappa, K, chi, m, M, alpha):
    theta, phi, dtheta, dphi = y
    a = alpha * M * gamma / chi
    b = (np.abs(m) * gamma**2 * H / chi - gamma**2 * H**2 + 2 * K * gamma**2 / chi)
    if m > 0:
        c = 2 * gamma * H - kappa * gamma**2 / chi
        ddtheta = -a * dtheta - b * theta - c * dphi
        ddphi = -a * dphi - b * phi + c * dtheta
    else:
        c = 2 * gamma * H + kappa * gamma**2 / chi
        ddtheta = -a * dtheta - b * theta + c * dphi
        ddphi = -a * dphi - b * phi - c * dtheta
    return [dtheta, dphi, ddtheta, ddphi]

def run_simulation(
        H_val: float,
        T_val: float,
        base_m: float,
        base_M: float,
        base_K: float,
        params: SimParams,
        t_max: float = simulation_time,
        num_points: int = 1001,
        method: str = 'DOP853',
        rtol: float = 1e-10,
        atol: float = 1e-12,
):
    # Применяем коэффициенты для подгона теории
    m_val   = params.m_scale * base_m
    M_val   = params.M_scale * base_M
    K_val   = params.k_scale * base_K
    chi_val = params.chi
    alpha   = params.alpha
    kappa   = m_val / gamma

    # Начальные условия (в радианах и рад/с)
    theta_initial = 0.0
    phi_initial = 0.0
    dtheta_initial = 0.0
    dphi_initial = (gamma**2) * (H_val + abs(m_val) / chi_val) * h_IFE * delta_t
                       
    y0 = [theta_initial, phi_initial, dtheta_initial, dphi_initial]
    t_eval = np.linspace(0, simulation_time, num_points)
    solution = solve_ivp(
        dynamics,
        (0.0, t_max),
        y0,
        t_eval=t_eval,
        args=(H_val, kappa, K_val, chi_val, m_val, M_val, alpha),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return solution.t, solution.y
