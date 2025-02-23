# constants.py
import numpy as np
from numba import njit, prange

# Основные параметры моделирования
N = 601  # 600 + 1
H_step = 10
H_lim = 4000
H_vals = np.arange(0, H_lim + 1, H_step)       # магнитное поле, Oe
T_vals = np.linspace(290, 350, N)                # температура, K

# Физические константы
gamma = 1.76e7       # рад/(с·Oe)
alpha = 3e-4
h_IFE = 7500         # Oe
delta_t = 250e-15    # с

# Загрузка предварительно сохранённых массивов
m_array = np.load('m_array.npy')
M_array = np.load('M_array.npy')
phi_amplitude = np.load('phi_amplitude.npy')
theta_amplitude = np.load('theta_amplitude.npy')

# Функции, зависящие от температуры
@njit(fastmath=True)
def K_T(T):
    return 0.19 * (T - 358)**2

@njit(fastmath=True)
def chi_T(T):
    return 4.2e-7 * np.abs(T - 358)

# Предвычисление meshgrid‑ов для H и T
H_mesh, T_mesh = np.meshgrid(H_vals, T_vals)
# Предполагается, что m_array и M_array соответствуют значениям T_vals
_, m_mesh = np.meshgrid(H_vals, m_array)
_, M_mesh = np.meshgrid(H_vals, M_array)
chi_mesh = chi_T(T_mesh)
K_mesh = K_T(T_mesh)

@njit(parallel=True, fastmath=True)
def compute_frequencies_numba(H_mesh, T_mesh, m_mesh, chi_mesh, K_mesh, gamma):
    rows, cols = H_mesh.shape
    f1_GHz = np.empty((rows, cols), dtype=np.float64)
    f2_GHz = np.empty((rows, cols), dtype=np.float64)
    for i in prange(rows):
        for j in range(cols):
            H_val = H_mesh[i, j]
            m_val = m_mesh[i, j]
            chi_val = chi_mesh[i, j]
            K_val = K_mesh[i, j]
            kappa_val = m_val / gamma
            if m_val > 0:
                common = (gamma**2 * H_val**2 +
                          2 * K_val * gamma**2 / chi_val +
                          np.abs(m_val) * H_val * gamma**2 / chi_val -
                          2 * kappa_val * gamma**3 * H_val / chi_val +
                          (kappa_val**2 * gamma**4) / (2 * chi_val**2))
                term = np.abs(2 * gamma * H_val - kappa_val * gamma**2 / chi_val)
                sqrt_term = np.sqrt((2 * K_val * gamma**2 / chi_val) +
                                    (np.abs(m_val) * H_val * gamma**2 / chi_val) -
                                    (kappa_val * gamma**3 * H_val / chi_val) +
                                    (kappa_val**2 * gamma**4) / (4 * chi_val**2))
            else:
                common = (gamma**2 * H_val**2 +
                          2 * K_val * gamma**2 / chi_val +
                          np.abs(m_val) * H_val * gamma**2 / chi_val +
                          kappa_val * gamma**3 * H_val / chi_val +
                          (kappa_val**2 * gamma**4) / (4 * chi_val**2))
                term = np.abs(2 * gamma * H_val + kappa_val * gamma**2 / chi_val)
                sqrt_term = np.sqrt((2 * K_val * gamma**2 / chi_val) +
                                    (np.abs(m_val) * H_val * gamma**2 / chi_val) +
                                    (kappa_val * gamma**3 * H_val / chi_val) +
                                    (kappa_val**2 * gamma**4) / (4 * chi_val**2))
            f1_sq = common + term * sqrt_term
            f2_sq = common - term * sqrt_term
            f1_GHz[i, j] = np.sqrt(f1_sq) / (2 * np.pi * 1e9)
            f2_GHz[i, j] = np.sqrt(f2_sq) / (2 * np.pi * 1e9)
    return f1_GHz, f2_GHz

# Вычисление частот с использованием оптимизированной функции
f1_GHz, f2_GHz = compute_frequencies_numba(H_mesh, T_mesh, m_mesh, chi_mesh, K_mesh, gamma)
