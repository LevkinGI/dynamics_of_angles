# constants.py
import numpy as np
from numba import njit, prange

# Исходные параметры (Материал 1)
H_step = 10
H_lim = 4000
H_vals = np.arange(0, H_lim + 1, H_step)
T_vals_1 = np.linspace(290, 350, 601)
T_vals_2 = np.linspace(290, 350, 61)
T_init = 293

gamma = 1.76e7              # рад/(с·Oe)
alpha_1 = 1e-4
alpha_2 = 1.7e-2
h_IFE = 7500                # Ое
delta_t = 250e-15           # с

# Функции, зависящие от температуры (Материал 1)
@njit
def K_T(T):
    return 0.19 * (T - 358)**2

@njit
def chi_T(T):
    return 4.2e-7 * np.abs(T - 358)

# Загрузка данных для материала 1
m_array_1 = np.load('m_array.npy')
M_array_1 = np.load('M_array.npy')
chi_array_1 = chi_T(T_vals_1)
K_array_1 = K_T(T_vals_1)
phi_amplitude = np.load('phi_amplitude.npy')
theta_amplitude = np.load('theta_amplitude.npy')

# Альтернативные данные для Материала 2
m_array_2 = np.load('m_array_2.npy')
M_array_2 = np.load('M_array_2.npy')
phi_amplitude_2 = np.load('phi_amplitude_2.npy')
theta_amplitude_2 = np.load('theta_amplitude_2.npy')

# Для материала 2 зависимости K(T) и chi(T) заменяем константами
chi_const = 3.7e-4
K_const = 13500
chi_array_2 = np.full_like(m_array_2, chi_const)
K_array_2 = np.full_like(m_array_2, K_const)

@njit(fastmath=True, cache=True)
def compute_frequencies(H_mesh, m_mesh, chi_mesh, K_mesh, gamma):
    kappa = m_mesh / gamma
    abs_m = np.abs(m_mesh)
    g2, g3, g4 = gamma**2, gamma**3, gamma**4
    H2 = H_mesh**2

    # Вычисления для m > 0
    sign = np.where(m_mesh < 0.0, -1.0, 1.0)
    common = (g2 * H2 +
              2 * K_mesh * g2 / chi_mesh +
              abs_m * H_mesh * g2 / chi_mesh -
              2 * sign * kappa * g3 * H_mesh / chi_mesh +
              (kappa**2 * g4) / (2 * chi_mesh**2))
    term   = np.abs(2 * gamma * H_mesh - sign * kappa * g2 / chi_mesh)
    sqrt_t = np.sqrt(2 * K_mesh * g2 / chi_mesh +
                     abs_m * H_mesh * g2 / chi_mesh -
                     sign * kappa * g3 * H_mesh / chi_mesh +
                     (kappa**2 * g4) / (4 * chi_mesh**2))
    
    f1 = np.sqrt(common + term * sqrt_t) / (2 * np.pi * 1e9)
    f2 = np.sqrt(common - term * sqrt_t) / (2 * np.pi * 1e9)
    return f1, f2

# Вычисление частот с использованием оптимизированной функции
# --- FeFe ---
H_mesh_1, m_mesh_1 = np.meshgrid(H_vals, m_array_1)
_, chi_mesh_1 = np.meshgrid(H_vals, chi_array_1)
_, K_mesh_1 = np.meshgrid(H_vals, K_array_1)

f1_GHz, f2_GHz = compute_frequencies(
        H_mesh_1,
        m_mesh_1,
        chi_mesh_1,
        K_mesh_1,
        gamma)

# --- GdFe ---
H_mesh_2, m_mesh_2 = np.meshgrid(H_vals, m_array_2)
_, chi_mesh_2 = np.meshgrid(H_vals, chi_array_2)
_, K_mesh_2 = np.meshgrid(H_vals, K_array_2)

# вектор по температуре, H – скаляр
@njit(cache=True, fastmath=True)
def compute_frequencies_H_fix(H, m_vec, chi_vec, K_vec, gamma):
    kappa_vec = m_vec / gamma
    abs_m = np.abs(m_vec)
    g2, g3, g4 = gamma**2, gamma**3, gamma**4
    H2 = H ** 2

    sign = np.where(m_vec < 0.0, -1.0, 1.0)
    common = (g2 * H2 +
              2 * K_vec * g2 / chi_vec +
              abs_m * H * g2 / chi_vec -
              2 * sign * kappa_vec * g3 * H / chi_vec +
              kappa_vec**2 * g4 / (2 * chi_vec**2))
    term = np.abs(2 * gamma * H - sign * kappa_vec * g2 / chi_vec)
    sqrt_t = np.sqrt(2 * K_vec * g2 / chi_vec +
                     abs_m * H * g2 / chi_vec -
                     sign * kappa_vec * g3 * H / chi_vec +
                     kappa_vec**2 * g4 / (4 * chi_vec**2))

    f1 = np.sqrt(common + term * sqrt_t) / (2*np.pi*1e9)
    f2 = np.sqrt(common - term * sqrt_t) / (2*np.pi*1e9)
    return f1, f2


# вектор по полю, T – скаляр
@njit(cache=True, fastmath=True)
def compute_frequencies_T_fix(H_vec, m, chi, K, gamma):
    kappa = m / gamma
    abs_m = abs(m)
    g2, g3, g4 = gamma**2, gamma**3, gamma**4
    H2   = H_vec*H_vec
    sign = 1.0 if m > 0 else -1.0

    common = (g2 * H2 +
              2 * K * g2 / chi +
              abs_m * H_vec * g2 / chi -
              2 * sign * kappa * g3 * H_vec / chi +
              kappa**2 * g4 / (2 * chi**2))
    term   = np.abs(2 * gamma * H_vec - sign * kappa * g2 / chi)
    sqrt_t = np.sqrt(2 * K * g2 / chi +
                     abs_m * H_vec * g2 / chi -
                     sign * kappa * g3 * H_vec / chi +
                     kappa**2 * g4 / (4 * chi**2))

    f1 = np.sqrt(common + term * sqrt_t) / (2*np.pi*1e9)
    f2 = np.sqrt(common - term * sqrt_t) / (2*np.pi*1e9)
    return f1, f2


__all__ = [
    # сетки и оси
    'H_vals', 'T_vals_1', 'T_vals_2', 'T_init',
    # исходные одномерные массивы (нужны графикам)
    'm_array_1', 'M_array_1', 'm_array_2', 'M_array_2',
    'chi_array_1', 'K_array_1', 'chi_array_2', 'K_array_2',
    # mesh
    'H_mesh_1', 'H_mesh_2',
    'm_mesh_1', 'chi_mesh_1', 'K_mesh_1',
    'm_mesh_2', 'chi_mesh_2', 'K_mesh_2',
    # физические константы
    'gamma', 'alpha_1', 'alpha_2', 'K_const', 'chi_const',
    'h_IFE', 'delta_t',
    # температурные функции
    'K_T', 'chi_T',
    # JIT-функция для частот
    'compute_frequencies', 'compute_frequencies_H_fix', 'compute_frequencies_T_fix'
    # частоты
    'f1_GHz', 'f2_GHz',
    # амплитуды
    'phi_amplitude', 'theta_amplitude', 'phi_amplitude_2', 'theta_amplitude_2',
]
