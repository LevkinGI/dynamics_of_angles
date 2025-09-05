# constants.py
import numpy as np
from numba import njit, prange

# Данные
T_293 = np.array([[1000, 1200, 1400, 1600, 1800, 2000],
                  [9.17440137,	9.423370201,	9.735918686,	10.01455683,	10.37994595,	10.5903492],
                  [29.35937721,	31.65155559,	30.2486405,	30.17815415,	29.94237192,	27.36357678]])
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

@njit(parallel=False, fastmath=False, cache=True)
def compute_frequencies(H_mesh, m_mesh, M_mesh, chi_mesh, K_mesh, gamma, alpha):
    nT, nH = H_mesh.shape
    total = nT * nH
    f1 = np.empty((nT, nH), np.float64)
    f2 = np.empty((nT, nH), np.float64)
    t1 = np.empty((nT, nH), np.float64)
    t2 = np.empty((nT, nH), np.float64)

    for idx in prange(total):
        i = idx // nH
        j = idx % nH
        H_ij   = H_mesh[i, j]
        m_ij   = m_mesh[i, j]
        M_ij   = M_mesh[i, j]
        chi_ij = chi_mesh[i, j]
        K_ij   = K_mesh[i, j]

        abs_m = np.abs(m_ij)

        w_H = gamma * H_ij
        w_0_sq = gamma**2 * 2 * K_ij / chi_ij
        w_KK = gamma * abs_m / chi_ij
      
        w_sq = w_0_sq + w_KK**2 / 4
        delta = w_H - w_KK / 2

        G = alpha * M_ij * gamma / (2 * chi_ij)
        W2 = w_sq - delta**2
        d_plus = delta - 1j * G
        d_minus = -delta - 1j * G

        w1 = np.sqrt(W2 + d_plus**2) + d_plus
        w2 = np.sqrt(W2 + d_minus**2) + d_minus
        w3 = -np.sqrt(W2 + d_plus**2) + d_plus
        w4 = -np.sqrt(W2 + d_minus**2) + d_minus

        roots = np.empty(4, np.complex128)
        roots[0] = w1; roots[1] = w2; roots[2] = w3; roots[3] = w4
        roots = roots[np.argsort(roots.real)[::-1]][:2]

        f1[i, j], t1[i, j] = roots[0].real / (2 * np.pi * 1e9), -1e9 / roots[0].imag
        f2[i, j], t2[i, j] = roots[1].real / (2 * np.pi * 1e9), -1e9 / roots[1].imag

    return (f1, t1), (f2, t2)

# @njit(parallel=False, fastmath=True, cache=True)
# def compute_frequencies(H_mesh, m_mesh, chi_mesh, K_mesh, gamma):
#     nT, nH = H_mesh.shape
#     total = nT * nH
#     f1 = np.empty((nT, nH), np.float64)
#     f2 = np.empty((nT, nH), np.float64)

#     g2, g3, g4 = gamma**2, gamma**3, gamma**4

#     for idx in prange(total):
#         i = idx // nH
#         j = idx % nH
#         H_ij   = H_mesh[i, j]
#         m_ij   = m_mesh[i, j]
#         chi_ij = chi_mesh[i, j]
#         K_ij   = K_mesh[i, j]

#         abs_m = np.abs(m_ij)
#         sign  = 1.0 if m_ij >= 0.0 else -1.0
#         kappa = m_ij / gamma
#         H2    = H_ij * H_ij

#         common = (
#             g2 * H2
#             + 2 * K_ij * g2 / chi_ij
#             + abs_m * H_ij * g2 / chi_ij
#             - 2 * sign * kappa * g3 * H_ij / chi_ij
#             + (kappa**2) * g4 / (2 * chi_ij**2)
#         )
#         term   = np.abs(2 * gamma * H_ij - sign * kappa * g2 / chi_ij)
#         sqrt_t = np.sqrt(
#             2 * K_ij * g2 / chi_ij
#             + abs_m * H_ij * g2 / chi_ij
#             - sign * kappa * g3 * H_ij / chi_ij
#             + (kappa**2) * g4 / (4 * chi_ij**2)
#         )

#         f1[i, j] = np.sqrt(common + term * sqrt_t) / (2 * np.pi * 1e9)
#         f2[i, j] = np.sqrt(common - term * sqrt_t) / (2 * np.pi * 1e9)

#     return f1, f2

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
@njit(parallel=True, fastmath=False, cache=True)
def compute_frequencies_H_fix(H, m_vec, M_vec, chi_vec, K_vec, gamma, alpha):
    n = m_vec.size
    f1 = np.empty(n, np.float64)
    t1 = np.empty(n, np.float64)
    f2 = np.empty(n, np.float64)
    t2 = np.empty(n, np.float64)

    w_H = gamma * H
    for i in prange(n):
        m   = m_vec[i]
        M   = M_vec[i]
        chi = chi_vec[i]
        K   = K_vec[i]

        abs_m = np.abs(m)

        w0_sq = gamma**2 * (2.0 * K / chi)
        w_KK  = gamma * abs_m / chi

        delta = w_H - w_KK / 2
        W2    = (w0_sq + w_KK**2 / 4) - delta**2

        G = alpha * M * gamma / (2.0 * chi)
        d_plus  =  delta - 1j * G
        d_minus = -delta - 1j * G

        w1 =  np.sqrt(W2 + d_plus**2) + d_plus
        w2 =  np.sqrt(W2 + d_minus**2) + d_minus
        w3 = -np.sqrt(W2 + d_plus**2) + d_plus
        w4 = -np.sqrt(W2 + d_minus**2) + d_minus

        roots = np.empty(4, np.complex128)
        roots[0] = w1; roots[1] = w2; roots[2] = w3; roots[3] = w4
        roots = roots[np.argsort(roots.real)[::-1]][:2]

        f1[i], t1[i] = roots[0].real / (2.0 * np.pi * 1e9), -1e9 / roots[0].imag
        f2[i], t2[i] = roots[1].real / (2.0 * np.pi * 1e9), -1e9 / roots[1].imag

    return (f1, t1), (f2, t2)

# @njit(parallel=True, cache=True, fastmath=True)
# def compute_frequencies_H_fix(H, m_vec, chi_vec, K_vec, gamma):
#     n = m_vec.size
#     f1 = np.empty(n, np.float64)
#     f2 = np.empty(n, np.float64)
#     g2, g3, g4 = gamma**2, gamma**3, gamma**4
#     H2 = H * H

#     # Параллельно проходим по всем i
#     for i in prange(n):
#         m_i   = m_vec[i]
#         chi_i = chi_vec[i]
#         K_i   = K_vec[i]
#         abs_m = np.abs(m_i)
#         sign  = -1.0 if m_i < 0.0 else 1.0
#         kappa = m_i / gamma

#         common = (
#             g2 * H2
#             + 2 * K_i * g2 / chi_i
#             + abs_m * H * g2 / chi_i
#             - 2 * sign * kappa * g3 * H / chi_i
#             + (kappa**2) * g4 / (2 * chi_i**2)
#         )
#         term   = np.abs(2 * gamma * H - sign * kappa * g2 / chi_i)
#         sqrt_t = np.sqrt(
#             2 * K_i * g2 / chi_i
#             + abs_m * H * g2 / chi_i
#             - sign * kappa * g3 * H / chi_i
#             + (kappa**2) * g4 / (4 * chi_i**2)
#         )

#         f1[i] = np.sqrt(common + term * sqrt_t) / (2 * np.pi * 1e9)
#         f2[i] = np.sqrt(common - term * sqrt_t) / (2 * np.pi * 1e9)

#     return f1, f2


# вектор по полю, T – скаляр
@njit(parallel=True, fastmath=False, cache=True)
def compute_frequencies_T_fix(H_vec, m, M, chi, K, gamma, alpha):
    n = H_vec.size
    f1 = np.empty(n, np.float64)
    t1 = np.empty(n, np.float64)
    f2 = np.empty(n, np.float64)
    t2 = np.empty(n, np.float64)

    abs_m = np.abs(m)

    # величины, не зависящие от H_j
    w0_sq = gamma**2 * (2.0 * K / chi)
    w_KK  = gamma * abs_m / chi
    G     = alpha * M * gamma / (2.0 * chi)
    for j in prange(n):
        H = H_vec[j]

        w_H  = gamma * H
        delta = w_H - 0.5 * w_KK
        W2    = (w0_sq + 0.25 * w_KK**2) - delta**2

        d_plus  =  delta - 1j * G
        d_minus = -delta - 1j * G

        w1 =  np.sqrt(W2 + d_plus**2) + d_plus
        w2 =  np.sqrt(W2 + d_minus**2) + d_minus
        w3 = -np.sqrt(W2 + d_plus**2) + d_plus
        w4 = -np.sqrt(W2 + d_minus**2) + d_minus

        roots = np.empty(4, np.complex128)
        roots[0] = w1; roots[1] = w2; roots[2] = w3; roots[3] = w4
        roots = roots[np.argsort(roots.real)[::-1]][:2]

        f1[j], t1[j] = roots[0].real / (2.0 * np.pi * 1e9), -1e9 / roots[0].imag
        f2[j], t2[j] = roots[1].real / (2.0 * np.pi * 1e9), -1e9 / roots[1].imag

    return (f1, t1), (f2, t2)

# @njit(parallel=True, cache=True, fastmath=True)
# def compute_frequencies_T_fix(H_vec, m, chi, K, gamma):
#     n = H_vec.size
#     f1 = np.empty(n, np.float64)
#     f2 = np.empty(n, np.float64)
#     g2, g3, g4 = gamma**2, gamma**3, gamma**4
#     sign = 1.0 if m > 0 else -1.0
#     kappa = m / gamma

#     # Параллельно по всем j
#     for j in prange(n):
#         H_j   = H_vec[j]
#         abs_m = np.abs(m)
#         H2    = H_j * H_j

#         common = (
#             g2 * H2
#             + 2 * K * g2 / chi
#             + abs_m * H_j * g2 / chi
#             - 2 * sign * kappa * g3 * H_j / chi
#             + (kappa**2) * g4 / (2 * chi**2)
#         )
#         term   = np.abs(2 * gamma * H_j - sign * kappa * g2 / chi)
#         sqrt_t = np.sqrt(
#             2 * K * g2 / chi
#             + abs_m * H_j * g2 / chi
#             - sign * kappa * g3 * H_j / chi
#             + (kappa**2) * g4 / (4 * chi**2)
#         )

#         f1[j] = np.sqrt(common + term * sqrt_t) / (2 * np.pi * 1e9)
#         f2[j] = np.sqrt(common - term * sqrt_t) / (2 * np.pi * 1e9)

#     return f1, f2

@njit(parallel=True, cache=True, fastmath=True)
def compute_phases(H_mesh, m_mesh, K_mesh, chi_mesh):
    nT, nH = H_mesh.shape
    total = nT * nH
    theta_0 = np.empty((nT, nH), np.float64)

    for idx in prange(total):
        i = idx // nH
        j = idx % nH
        H_ij = H_mesh[i, j]
        if H_ij == 0:
            theta_0[i, j] = np.nan
            continue
        m_ij   = m_mesh[i, j]
        abs_m = np.abs(m_ij)
        chi_ij = chi_mesh[i, j]
        K_ij   = K_mesh[i, j]

        m_cr = chi_ij * H_ij + (2 * K_ij) / H_ij
        theta_0[i, j] = 0.0 if abs_m > m_cr else np.arccos(abs_m / m_cr)
      
    return theta_0


__all__ = [
    # сетки и оси
    'H_vals', 'T_vals_1', 'T_vals_2', 'T_init',
    # исходные одномерные массивы (нужны графикам)
    'm_array_1', 'M_array_1', 'm_array_2', 'M_array_2',
    'chi_array_1', 'K_array_1', 'chi_array_2', 'K_array_2',
    # mesh
    'H_mesh_1', 'H_mesh_2',
    'm_mesh_1', 'M_mesh_1', 'chi_mesh_1', 'K_mesh_1',
    'm_mesh_2', 'M_mesh_2', 'chi_mesh_2', 'K_mesh_2',
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
    'T_293', 'H_1000',
]
