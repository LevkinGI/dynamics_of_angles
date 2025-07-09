# constants.py
import numpy as np
from numba import njit

# Исходные параметры (Материал 1)
H_step = 10
H_lim = 4000
H_vals = np.arange(0, H_lim + 1, H_step)
T_vals_1 = np.linspace(290, 350, 601)
T_vals_2 = np.linspace(290, 350, 61)

gamma = 1.76e7              # рад/(с·Oe)
alpha_1 = 1e-4
alpha_2 = 1.7e-2
h_IFE = 7500                # Ое
delta_t = 250e-15           # с
simulation_time = 0.3e-9    # с

# Загрузка данных для материала 1
m_array = np.load('m_array.npy')
M_array = np.load('M_array.npy')
phi_amplitude = np.load('phi_amplitude.npy')
theta_amplitude = np.load('theta_amplitude.npy')

# Функции, зависящие от температуры (Материал 1)
@njit
def K_T(T):
    return 0.19 * (T - 358)**2

@njit
def chi_T(T):
    return 4.2e-7 * np.abs(T - 358)

# Альтернативные данные для Материала 2
m_array_2 = np.load('m_array_2.npy')
M_array_2 = np.load('M_array_2.npy')
phi_amplitude_2 = np.load('phi_amplitude_2.npy')
theta_amplitude_2 = np.load('theta_amplitude_2.npy')

# Для материала 2 зависимости K(T) и chi(T) заменяем константами
K_const = 13500
chi_const = 3.7e-4

H_mesh_1, T_mesh_1 = np.meshgrid(H_vals, T_vals_1)
H_mesh_2, T_mesh_2 = np.meshgrid(H_vals, T_vals_2)

# Предвычисление meshgrid’ов и частот для материала 1
_, m_mesh_1 = np.meshgrid(H_vals, m_array)
_, M_mesh_1 = np.meshgrid(H_vals, M_array)
chi_mesh_1 = chi_T(T_mesh_1)
K_mesh_1 = K_T(T_mesh_1)

# Предвычисление meshgrid’ов и частот для материала 2
_, m_mesh_2 = np.meshgrid(H_vals, m_array_2)
_, M_mesh_2 = np.meshgrid(H_vals, M_array_2)
chi_mesh_2 = chi_const * np.ones(m_mesh_2.shape)
K_mesh_2 = K_const * np.ones(m_mesh_2.shape)

from numba import njit, prange

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
                sqrt_term = np.sqrt(2 * K_val * gamma**2 / chi_val +
                                    np.abs(m_val) * H_val * gamma**2 / chi_val -
                                    kappa_val * gamma**3 * H_val / chi_val +
                                    (kappa_val**2 * gamma**4) / (4 * chi_val**2))
            else:
                common = (gamma**2 * H_val**2 +
                          2 * K_val * gamma**2 / chi_val +
                          np.abs(m_val) * H_val * gamma**2 / chi_val +
                          2 * kappa_val * gamma**3 * H_val / chi_val +
                          (kappa_val**2 * gamma**4) / (2 * chi_val**2))
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
# --- FeFe ---
f1_GHz_1, f2_GHz_1 = compute_frequencies_numba(
        H_mesh_1, T_mesh_1,
        m_mesh_1,          # вместо m_array
        np.full_like(m_mesh_1, chi_T(T_mesh_1)),  # χ_скаляр → 2D
        K_mesh_1,
        gamma)

# --- GdFe ---
f1_GHz_2, f2_GHz_2 = compute_frequencies_numba(
        H_mesh_2, T_mesh_2,
        m_mesh_2,
        np.full_like(m_mesh_2, chi_const),
        K_mesh_2,
        gamma)

__all__ = [
    # сетки и оси
    'H_vals', 'T_vals_1', 'T_vals_2',
    'H_mesh_1', 'H_mesh_2', 'T_mesh_1', 'T_mesh_2',
    # массивы m, M, K
    'm_mesh_1', 'm_mesh_2', 'M_mesh_1', 'M_mesh_2',
    'K_mesh_1', 'K_mesh_2',
    # частоты (будут потом перезаписываться)
    'f1_GHz_1', 'f2_GHz_1', 'f1_GHz_2', 'f2_GHz_2',
    # исходные одномерные массивы (нужны графикам)
    'm_array', 'M_array', 'm_array_2', 'M_array_2',
    # физические константы
    'gamma', 'alpha_1', 'alpha_2', 'K_const', 'chi_const',
    'h_IFE', 'delta_t',
    # температурные функции
    'K_T', 'chi_T',
    # JIT-функция для частот
    'compute_frequencies_numba',
]
