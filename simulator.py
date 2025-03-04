# simulator.py
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from constants import gamma, alpha, h_IFE, delta_t

@njit(fastmath=True)
def dynamics(t, y, H, T, m, M, chi, K, kappa, gamma, alpha):
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

def run_simulation(H, T, m, M, chi, K, kappa, simulation_time=0.3e-9, num_points=1001,
                   method='DOP853', rtol=1e-10, atol=1e-12):
    # Начальные условия (в радианах и рад/с)
    theta_initial = 0.0
    phi_initial = 0.0
    dtheta_initial = 0.0
    dphi_initial = gamma**2 * (H + np.abs(m)/chi) * h_IFE * delta_t
                       
    y0 = [theta_initial, phi_initial, dtheta_initial, dphi_initial]
    t_eval = np.linspace(0, simulation_time, num_points)
    solution = solve_ivp(
        dynamics,
        (0, simulation_time),
        y0,
        args=(H, T, m, M, chi, K, kappa, gamma, alpha),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol
    )
    return solution.t, solution.y
