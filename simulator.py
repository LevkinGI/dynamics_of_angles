# simulator.py
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from constants import gamma, h_IFE, delta_t

def _dynamics_factory(a: float, b: float, c: float, sign: int):
    @njit(cache=True, fastmath=True)
    def dynamics_const(t, y):
        theta, phi, dtheta, dphi = y
        ddtheta = -a * dtheta - b * theta - sign * c * dphi
        ddphi = -a * dphi - b * phi + sign * c * dtheta
        return (dtheta, dphi, ddtheta, ddphi)

    # ➜ вернём Python-обёртку — она вызывает уже скомпилированный код
    return dynamics_const.py_func

# @njit(fastmath=True)
# def dynamics(t, y, H, m, M, K, chi, alpha, kappa):
#     theta, phi, dtheta, dphi = y
#     a = alpha * M * gamma / chi
#     b = (np.abs(m) * gamma**2 * H / chi - gamma**2 * H**2 + 2 * K * gamma**2 / chi)
#     if m > 0:
#         c = 2 * gamma * H - kappa * gamma**2 / chi
#         ddtheta = -a * dtheta - b * theta - c * dphi
#         ddphi = -a * dphi - b * phi + c * dtheta
#     else:
#         c = 2 * gamma * H + kappa * gamma**2 / chi
#         ddtheta = -a * dtheta - b * theta + c * dphi
#         ddphi = -a * dphi - b * phi - c * dtheta
#     return (dtheta, dphi, ddtheta, ddphi)

def run_simulation(
        H_val: float,
        T_val: float,
        m_val: float,
        M_val: float,
        K_val: float,
        chi_val: float,
        alpha: float,
        kappa: float,
        simulation_time: float = 0.3e-9,
        num_points: int = 1001,
        method: str = 'DOP853',
        rtol: float = 1e-10,
        atol: float = 1e-12,
):
    # Начальные условия (в радианах и рад/с)
    theta_initial = 0.0
    phi_initial = 0.0
    dtheta_initial = 0.0
    dphi_initial = (gamma**2) * (H_val + abs(m_val) / chi_val) * h_IFE * delta_t

    a = alpha * M_val * gamma / chi_val
    b = (np.abs(m_val) * gamma**2 * H_val / chi_val - gamma**2 * H_val**2 + 2 * K_val * gamma**2 / chi_val)
    sign = 1 if m_val > 0 else -1
    c = 2 * gamma * H_val - sign * kappa * gamma**2 / chi_val
    dynamics = _dynamics_factory(a, b, c, sign)
                       
    y0 = [theta_initial, phi_initial, dtheta_initial, dphi_initial]
    t_eval = np.linspace(0, simulation_time, num_points)
    sol = solve_ivp(
        dynamics,
        (0.0, simulation_time),
        y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y
