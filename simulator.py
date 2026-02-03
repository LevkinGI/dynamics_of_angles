# simulator.py
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from constants import gamma, h_IFE, delta_t, chi_func

def _dynamics_factory(a: float, b: float, c: float, sign: int):
    @njit(cache=True, fastmath=True)
    def dynamics_const(t, y):
        theta, phi, dtheta, dphi = y
        ddtheta = -a * dtheta - b * theta - sign * c * dphi
        ddphi = -a * dphi - b * phi + sign * c * dtheta
        return (dtheta, dphi, ddtheta, ddphi)

    # ➜ вернём Python-обёртку — она вызывает уже скомпилированный код
    return dynamics_const.py_func

@njit(cache=True, fastmath=True)
def calc_coef(H: float, m: float, M: float, K: float,
              chi:float, alpha: float, kappa: float):
    a = alpha * M * gamma / chi
    b = (np.abs(m) * gamma**2 * H / chi - gamma**2 * H**2 + 2 * K * gamma**2 / chi)
    sign = 1 if m > 0 else -1
    c = 2 * gamma * H - sign * kappa * gamma**2 / chi

    return (a, b, c, sign)

# def run_simulation(
#         H: float,
#         m: float,
#         M: float,
#         K: float,
#         alpha: float,
#         lam: float,
#         kappa: float,
#         simulation_time: float = 0.3e-9,
#         num_points: int = 1001,
#         method: str = 'DOP853',
#         rtol: float = 1e-10,
#         atol: float = 1e-12,
# ):
#     chi = chi_func(m, M, lam)
#     # Начальные условия (в радианах и рад/с)
#     theta_initial = 0.0
#     phi_initial = 0.0
#     dtheta_initial = 0.0
#     dphi_initial = (gamma**2) * (H + abs(m) / chi) * h_IFE * delta_t

#     a, b, c, sign = calc_coef(H, m, M, K, chi, alpha, kappa)
#     dynamics = _dynamics_factory(a, b, c, sign)
                       
#     y0 = [theta_initial, phi_initial, dtheta_initial, dphi_initial]
#     t_eval = np.linspace(0, simulation_time, num_points)
#     sol = solve_ivp(
#         dynamics,
#         (0.0, simulation_time),
#         y0,
#         t_eval=t_eval,
#         method=method,
#         rtol=rtol,
#         atol=atol,
#     )
#     if not sol.success:
#         raise RuntimeError(sol.message)
#     return sol.t, sol.y

def run_simulation(
        H: float,
        m: float,
        M: float,
        K: float,
        alpha: float,
        lam: float,
        kappa: float,
        simulation_time: float = 0.3e-9,
        num_points: int = 1001,
        method: str = 'DOP853',
        rtol: float = 1e-10,
        atol: float = 1e-12,
        two_pulses: bool = True,
        t_pulse2: float = 0.0,
):
    chi = chi_func(m, M, lam)
    knock = (gamma**2) * (H + abs(m) / chi) * h_IFE * delta_t

    # Начальные условия (в радианах и рад/с)
    theta_initial = 0.0
    phi_initial = 0.0
    dtheta_initial = 0.0
    dphi_initial = knock

    a, b, c, sign = calc_coef(H, m, M, K, chi, alpha, kappa)
    dynamics = _dynamics_factory(a, b, c, sign)

    y0 = np.array([theta_initial, phi_initial, dtheta_initial, dphi_initial], dtype=float)

    # t_eval общий
    t_eval = np.linspace(0.0, simulation_time, num_points)

    # Если второй импульс выключен или вне интервала — обычная интеграция
    if (not two_pulses) or (t_pulse2 is None) or (t_pulse2 <= 0.0) or (t_pulse2 >= simulation_time):
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

    # --- ДВА УЧАСТКА: [0, t_pulse2] и [t_pulse2, T] ---
    # Разбиваем t_eval так, чтобы точка t_pulse2 была в сетке
    # (если её нет — добавим вручную)
    left_mask = t_eval < t_pulse2
    right_mask = t_eval > t_pulse2

    t_left = t_eval[left_mask]
    t_right = t_eval[right_mask]

    # гарантируем наличие точки t_pulse2
    if (t_left.size == 0) or (t_left[-1] != t_pulse2):
        t_left = np.concatenate([t_left, np.array([t_pulse2])])

    # 1) интегрируем до t_pulse2
    sol1 = solve_ivp(
        dynamics,
        (0.0, float(t_pulse2)),
        y0,
        t_eval=t_left,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol1.success:
        raise RuntimeError(sol1.message)

    y_at_pulse = sol1.y[:, -1].copy()

    # 2) второй импульс
    y_at_pulse[3] += knock  # dtheta += knock

    # если правой части нет — всё закончили
    if t_right.size == 0:
        return sol1.t, sol1.y

    # 3) интегрируем от t_pulse2 до конца
    sol2 = solve_ivp(
        dynamics,
        (float(t_pulse2), float(simulation_time)),
        y_at_pulse,
        t_eval=t_right,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol2.success:
        raise RuntimeError(sol2.message)

    # склейка результатов
    t = np.concatenate([sol1.t, sol2.t])
    y = np.concatenate([sol1.y, sol2.y], axis=1)
    return t, y
