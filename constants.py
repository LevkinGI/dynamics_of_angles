# constants.py
import numpy as np
from numba import njit, prange

# Данные
T_293 = np.array([[1000, 1200, 1400, 1600, 1800, 2000],
                  [9.17440137,	9.423370201,	9.735918686,	10.01455683,	10.37994595,	10.5903492],
                  [29.35937721,	31.65155559,	30.2486405,	30.17815415,	29.94237192,	27.36357678]])
T_310 = np.array([[1000, 1200, 1400, 1600, 1800],
                  [9.86247161,	9.971845104,	10.68972025,	10.63944219,	10.35700201],
                  [17.09233677,	20.56842378,	19.38801173,	18.72971443,	22.76495312]])
T_323 = np.array([[1000, 1200, 1400, 1600, 1800],
                  [10.32385875,	10.62451462,	10.95407703,	10.69516529,	10.92184819],
                  [18.12080539,	69.36782127,	31.5,	38.92226665,	30.48632465]])
H_1000 = np.array([[298, 302, 308, 313, 318, 323, 328, 333],
                   [9.361751347, 9.588230642, 10.05690076, 10.44679623, 7.21359914, 8.191941342, 5.847037888, 3.470132708],
                   [25.10904523, 23.83906037, 18.5711219, 17.04232042, 10.56613241, 10.84025321, 11.53883964, 11.10824405]])

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
@njit(cache=True)
def K_T(T):
    return 0.525 * (T - 370)**2

@njit(cache=True)
def chi_T(T):
    return 4.2e-7 * np.abs(T - 358)

# Загрузка данных для материала 1
m_array_1 = np.load('m_array_18.07.2025.npy')
M_array_1 = np.load('M_array_18.07.2025.npy')
chi_array_1 = chi_T(T_vals_1) if False else np.full_like(m_array_1, 8e-5)
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

def compute_frequencies(H_mesh, m_mesh, M_mesh, chi_mesh, K_mesh, gamma, alpha):
    abs_m = np.abs(m_mesh)

    w_H = gamma * H_mesh
    w_0_sq = gamma**2 * 2 * K_mesh / chi_mesh
    w_KK = gamma * abs_m / chi_mesh

    w_sq = w_0_sq + w_KK**2 / 4
    delta = w_H - w_KK / 2

    G = alpha * M_mesh * gamma / (2 * chi_mesh)
    W2 = w_sq - delta**2
    d_plus = delta - 1j * G
    d_minus = -delta - 1j * G

    w1 = np.sqrt(W2 + d_plus**2) + d_plus
    w2 = np.sqrt(W2 + d_minus**2) + d_minus
    w3 = -np.sqrt(W2 + d_plus**2) + d_plus
    w4 = -np.sqrt(W2 + d_minus**2) + d_minus

    roots = np.stack((w1, w2, w3, w4), axis=-1)
    sorted_indices = np.argsort(roots.real, axis=-1)[:, :, ::-1]
    sorted_roots = np.take_along_axis(roots, sorted_indices, axis=-1)

    f1, t1 = sorted_roots.real[:, :, 0] / (2 * np.pi * 1e9), -1e9 / sorted_roots.imag[:, :, 0]
    f2, t2 = sorted_roots.real[:, :, 1] / (2 * np.pi * 1e9), -1e9 / sorted_roots.imag[:, :, 1]
    return (f1, t1), (f2, t2)

# Вычисление частот
# --- FeFe ---
H_mesh_1, m_mesh_1 = np.meshgrid(H_vals, m_array_1)
_, M_mesh_1 = np.meshgrid(H_vals, M_array_1)
_, chi_mesh_1 = np.meshgrid(H_vals, chi_array_1)
_, K_mesh_1 = np.meshgrid(H_vals, K_array_1)

(f1_GHz, _), (f2_GHz, _) = compute_frequencies(
        H_mesh_1,
        m_mesh_1,
        M_mesh_1,
        chi_mesh_1,
        K_mesh_1,
        gamma,
        alpha_1)

# --- GdFe ---
H_mesh_2, m_mesh_2 = np.meshgrid(H_vals, m_array_2)
_, M_mesh_2 = np.meshgrid(H_vals, M_array_2)
_, chi_mesh_2 = np.meshgrid(H_vals, chi_array_2)
_, K_mesh_2 = np.meshgrid(H_vals, K_array_2)

# вектор по температуре, H – скаляр
def compute_frequencies_H_fix(H, m_vec, M_vec, chi_vec, K_vec, gamma, alpha):
    abs_m = np.abs(m_vec)

    w_H  = gamma * H
    w0_sq = gamma**2 * (2.0 * K_vec / chi_vec)
    w_KK  = gamma * abs_m / chi_vec

    delta = w_H - w_KK / 2
    W2    = (w0_sq + w_KK**2 / 4) - delta**2

    G = alpha * M_vec * gamma / (2.0 * chi_vec)
    d_plus  =  delta - 1j * G
    d_minus = -delta - 1j * G

    w1 =  np.sqrt(W2 + d_plus**2) + d_plus
    w2 =  np.sqrt(W2 + d_minus**2) + d_minus
    w3 = -np.sqrt(W2 + d_plus**2) + d_plus
    w4 = -np.sqrt(W2 + d_minus**2) + d_minus

    roots = np.stack((w1, w2, w3, w4), axis=-1)
    sorted_indices = np.argsort(roots.real, axis=-1)[:, ::-1]
    sorted_roots = np.take_along_axis(roots, sorted_indices, axis=-1)

    f1, t1 = sorted_roots.real[:, 0] / (2 * np.pi * 1e9), -1e9 / sorted_roots.imag[:, 0]
    f2, t2 = sorted_roots.real[:, 1] / (2 * np.pi * 1e9), -1e9 / sorted_roots.imag[:, 1]

    return (f1, t1), (f2, t2)

# вектор по полю, T – скаляр
def compute_frequencies_T_fix(H_vec, m, M, chi, K, gamma, alpha):
    abs_m = np.abs(m)
    
    w_H   = gamma * H_vec
    w0_sq = gamma**2 * (2.0 * K / chi)
    w_KK  = gamma * abs_m / chi
    G     = alpha * M * gamma / (2.0 * chi)

    delta = w_H - 0.5 * w_KK
    W2    = (w0_sq + 0.25 * w_KK**2) - delta**2

    d_plus  =  delta - 1j * G
    d_minus = -delta - 1j * G

    w1 =  np.sqrt(W2 + d_plus**2) + d_plus
    w2 =  np.sqrt(W2 + d_minus**2) + d_minus
    w3 = -np.sqrt(W2 + d_plus**2) + d_plus
    w4 = -np.sqrt(W2 + d_minus**2) + d_minus

    roots = np.stack((w1, w2, w3, w4), axis=-1)
    sorted_indices = np.argsort(roots.real, axis=-1)[:, ::-1]
    sorted_roots = np.take_along_axis(roots, sorted_indices, axis=-1)

    f1, t1 = sorted_roots.real[:, 0] / (2 * np.pi * 1e9), -1e9 / sorted_roots.imag[:, 0]
    f2, t2 = sorted_roots.real[:, 1] / (2 * np.pi * 1e9), -1e9 / sorted_roots.imag[:, 1]

    return (f1, t1), (f2, t2)

def compute_phases(H_mesh, m_mesh, K_mesh, chi_mesh):
    abs_m = np.abs(m_mesh)
    m_cr = chi_mesh * H_mesh + (2 * K_mesh) / H_mesh
    theta_0 = np.where(H_mesh==0, np.nan, np.where(abs_m > m_cr, 0.0, np.arccos(abs_m / m_cr)))
      
    return theta_0


__all__ = [
    # сетки и оси
    'H_vals', 'T_vals_1', 'T_vals_2', 'T_init',
    # исходные одномерные массивы (нужны графикам)
    'm_array_1', 'M_array_1', 'm_array_2', 'M_array_2',
    'chi_array_1', 'K_array_1', 'chi_array_2', 'K_array_2',
    # физические константы
    'gamma', 'alpha_1', 'alpha_2', 'K_const', 'chi_const',
    'h_IFE', 'delta_t',
    # JIT-функция для частот
    'compute_frequencies', 'compute_phases',
    'compute_frequencies_H_fix', 'compute_frequencies_T_fix',
    # частоты
    'f1_GHz', 'f2_GHz',
    # амплитуды
    'phi_amplitude', 'theta_amplitude', 'phi_amplitude_2', 'theta_amplitude_2',
    # данные
    'T_293', 'T_310', 'T_323', 'H_1000',
]
