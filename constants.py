# constants.py
import numpy as np
from numba import njit, prange
from typing import Iterable

# Данные
T_293 = np.array([[1000, 1200, 1400, 1600, 1800, 2000],
                  [9.17440137,  9.423370201,  9.735918686,  10.01455683,  10.37994595,  10.5903492],
                  [29.35937721, 31.65155559,  30.2486405,   30.17815415,  29.94237192,  27.36357678],
                  [0.2326714,   0.2440502,    0.2528819,    0.2692589,	  0.2764174,    0.3013753],
                  [0.01385995,  0.01490037,   0.01370499,   0.01470922,   0.0150656,    0.02085652]])
T_298 = np.array([[300,	400,	500,	600,	700,	800,	900,	1000,	1100,	1200,	1300,	1400,	1500,	1600,	1700],
                  [7.961283724,	8.112452867,	8.334947049,	8.542386482,	8.818722633,	9.009857803,	9.243014414,	9.326656364,	9.586846435,	9.833754324,	10.00096132,	10.26695048,	10.48170457,	10.66891249,	10.86957824],
                  [35.9503266,	35.30640379,	36.2904306,	  36.23461139,	35.3250771,	  35.73303518,	35.64553591,	35.57496574,	34.12527117,	34.01280056,	35.02376791,	34.89270032,	34.05328875,	33.98951909,	34.74480867],
                  [0.387293092,	0.4689481,  	0.395373606,	0.367162912,	0.444376284,	0.44448003,  	0.39913016,  	0.417542102,	0.355354041,	0.364708072,	0.42127274,  	0.399316325,	0.433358941,	0.420070869,	0.44874486],
                  [0.027872295,	0.02828025,  	0.027146981,	0.02552049,  	0.028944025,	0.027859766,	0.025893264,	0.020068313,	0.024111124,	0.022539052,	0.023337985,	0.020271262,	0.021780195,	0.021205039,	0.022513286]])
T_308 =  np.array([[300,	400,	500,	600,	700,	800,	900,	1000,	1100,	1200,	1300,	1400,	1500,	1600,	1700],
                   [8.937549551,	9.183303801,	9.436465239,	9.672472021,	9.799494481,	9.888207347,	10.0839591,	  10.22131939,	10.3813667,  	10.57739003,	10.7075884,  	10.90387457,	10.95307609,	11.28980085,	11.35866279],
                   [22.7019311,  	22.67961303,	22.00522988,	21.67647155,	22.41350342,	22.24309327,	21.6154545,	  21.49189617,	21.55743187,	21.23982667,	21.04952922,	20.04489697,	23.64152474,	20.05336376,	25.41081746],
                   [0.189330769,	0.205763739,	0.209861742,	0.186501848,	0.23591113,  	0.243050325,	0.240669259,  0.240415501,	0.268157207,	0.279112256,	0.292817478,	0.342335456,	0.285971372,	0.284999857,	0.252987193],
                   [0.029700689,	0.027460218,	0.02700383,  	0.026121174,	0.022535344,	0.02116828,  	0.022816963,	0.024243045,	0.02556581,  	0.021633368,	0.023212102,	0.017383083,	0.024463791,	0.020187841,	0.028001553]])
T_310 = np.array([[1000, 1200, 1400, 1600, 1800],
                  [9.608579,    10.1564,	    10.48156,     10.75176,	    10.5243],
                  [19.53544,    20.56842378,  19.6038,      18.23266,	    22.76495312],
                  [0.1791479,   0.2147599,    0.1493049,    0.1797168,    0.2107221],
                  [0.025845326, 0.02878221,   0.05083024,   0.03752954,   None]])
T_320 = np.array([[300,	400,	500,	600,	700,	800,	900,	1000,	1100,	1200,	1300,	1400,	1500,	1600,	1700],
                  [10.72802486,	11.20301435,	11.57000741,	10.24082858,	11.93735544,	12.48866159,	11.16335263,	12.95603949,	12.16214859,	10.50310187,	9.121661974,	10.64765806,	8.684410975,	8.305370361,	8.744966258],
                  [16.3200493,	20.49070545,	14.82584226,	11.55677015,	13.10077717,	14.24688068,	12.01294889,	13.30030656,	12.68035458,	11.685925,  	12.23943664,	11.78317385,	12.18310692,	12.35017465,	11.97393299],
                  [0.090305018,	0.085822521,	0.08579766,  	0.061286937,	0.044183669,	0.07908596,  	0.092696218,	0.070508064,	0.123427243,	0.08249945,  	0.050000001,	0.072798425,	0.066971841,	0.087370109,	0.059157179],
                  [0.08534056,	0.229407668,	0.02441555,  	0.281307108,	0.079582647,	0.058288813,	0.263201365,	0.059819846,	0.02618461,  	0.240384336,	0.238075321,	0.349512356,	0.227642136,	0.212546243,	0.308562098]])
T_323 = np.array([[1000, 1200, 1400, 1600, 1800],
                  [8.0366,      6.103743,     1.262356544,  3.568812289,  3.787543515],
                  [10.16,       10.49205139,  11.05093022,  10.84827952,  11.39313551],
                  [0.1512544,   0.1455957,    2.86294,      0.8830667,    0.1959008],
                  [0.1470505,   0.2757782,    0.1217176,    0.110712,     0.1518173]])
H_1000 = np.array([[298,	303,	308,	313,	318,	323,	328,	333],
                   [9.282562466,	9.807333871,	10.09330134,	10.61328112,	7.105452097,	6.857629475,	4.791499676,	3.293120638],
                   [36.07808758,	26.99033281,	22.16470557,	20.76184916,	11.50559268,	10.884979,	  12.41093083,	13.96955417],
                   [0.331578945,	0.242661953,	0.222036407,	0.163352103,	0.077923117,	0.151206055,	0.063442916,	0.05],
                   [0.02789031,	  0.021279605,	0.029382366,	0.044387055,	0.128465789,	0.077210813,	0.144562978,	0.5]])
H_1700 = np.array([[298,	303,	308,	313,	318,	323,	328,	333],
                   [10.64395404,	11.08859404,	11.33419432,	11.76530759,	9.800271063,	8.937050334,	4.076657625,	2.653114246],
                   [37.43165126,	29.22627049,	23.1846905,  	17.43323314,	11.94522728,	13.48271963,	11.64730936,	15.26407752],
                   [0.818144961,	0.484518225,	0.350349669,	0.215105189,	0.072835581,	0.119587347,	0.050000003,	0.048582682],
                   [0.030453149,	0.027546426,	0.020729218,	0.033482177,	0.295981413,	0.077621878,	0.082861392,	0.055900702]])

# Исходные параметры (Материал 1)
H_step = 10
H_lim = 2000
H_vals = np.arange(0, H_lim + 1, H_step)
T_vals_1 = np.linspace(290, 350, 601)
T_vals_2 = np.linspace(290, 350, 61)
T_init = 298

gamma = 1.76e7              # рад/(с·Oe)
lam_1 = 12500
lam_2 = 10000 # Заглушка!
alpha_1 = 1e-3
alpha_2 = 1.7e-2
h_IFE = 7500                # Ое
delta_t = 250e-15           # с

# Материал 1
@njit(cache=False, fastmath=True)
def K_T(T: Iterable[float] | float) -> np.ndarray:
    """Анизотропия как функция температуры."""
    return 0.522 * (T - 370.0) ** 2

@njit(cache=False, fastmath=True)
def chi_func(m: np.ndarray | float, M: np.ndarray | float, lam: float) -> np.ndarray | float:
    """Вычисление магнитной восприимчивости."""
    denom = 1.0 - (m**2) / (M**2 + 1e-16)
    denom = np.where(denom == 0, np.nan, denom)
    return 1.0 / (lam * denom)

# Загрузка данных для материала 1
m_array_1 = np.load('m_array_18.07.2025.npy')
M_array_1 = np.load('M_array_18.07.2025.npy')
K_array_1 = K_T(T_vals_1)

# Альтернативные данные для Материала 2
m_array_2 = np.load('m_array_2.npy')
M_array_2 = np.load('M_array_2.npy')

# Для материала 2 зависимости K(T) и chi(T) заменяем константами
K_const = 13500
K_array_2 = np.full_like(m_array_2, K_const)

def compute_frequencies(H_vals, m_array, M_array, K_array, gamma, alpha, lam):
    H_mesh, m_mesh = np.meshgrid(H_vals, m_array)
    _, M_mesh = np.meshgrid(H_vals, M_array)
    _, K_mesh = np.meshgrid(H_vals, K_array)
    chi_mesh = chi_func(m_mesh, M_mesh, lam)
  
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

    f1, f2 = np.where(delta <= 0, f1, f2), np.where(delta <= 0, f2, f1)
    t1, t2 = np.where(delta <= 0, t1, t2), np.where(delta <= 0, t2, t1)
    
    return (f1, t1), (f2, t2)

# вектор по температуре, H – скаляр
def compute_frequencies_H_fix(H, m_vec, M_vec, K_vec, gamma, alpha, lam):
    chi_vec = chi_func(m_vec, M_vec, lam)
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

    f1, f2 = np.where(delta <= 0, f1, f2), np.where(delta <= 0, f2, f1)
    t1, t2 = np.where(delta <= 0, t1, t2), np.where(delta <= 0, t2, t1)

    return (f1, t1), (f2, t2)

# вектор по полю, T – скаляр
def compute_frequencies_T_fix(H_vec, m, M, K, gamma, alpha, lam):
    chi = chi_func(m, M, lam)
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

    f1, f2 = np.where(delta <= 0, f1, f2), np.where(delta <= 0, f2, f1)
    t1, t2 = np.where(delta <= 0, t1, t2), np.where(delta <= 0, t2, t1)

    if K < K_T(333): f1, f2, t1, t2 = f2, f1, t2, t1

    return (f1, t1), (f2, t2)

def compute_phases(m_array, M_array, K_array, lam):
    H_vals = np.arange(0, 4001, 25)
    H_mesh, m_mesh = np.meshgrid(H_vals, m_array)
    _, M_mesh = np.meshgrid(H_vals, M_array)
    _, K_mesh = np.meshgrid(H_vals, K_array)
    chi_mesh = chi_func(m_mesh, M_mesh, lam)
  
    abs_m = np.abs(m_mesh)
    m_cr = np.where(H_mesh==0, np.nan, chi_mesh * H_mesh + (2 * K_mesh) / H_mesh)
    theta_0 = np.where(H_mesh==0, np.nan, np.where(abs_m > m_cr, 0.0, np.arccos(abs_m / m_cr)))
      
    return theta_0


__all__ = [
    # сетки и оси
    'H_vals', 'T_vals_1', 'T_vals_2', 'T_init',
    # исходные одномерные массивы
    'm_array_1', 'M_array_1', 'm_array_2', 'M_array_2',
    'K_array_1', 'K_array_2',
    # физические константы
    'gamma', 'lam_1', 'lam_2', 'alpha_1', 'alpha_2',
    'K_const', 'h_IFE', 'delta_t',
    # JIT-функция для частот
    'compute_frequencies', 'compute_phases',
    'compute_frequencies_H_fix', 'compute_frequencies_T_fix',
    # данные
    'T_293', 'T_298', 'T_308', 'T_310', 'T_320', 'T_323',
    'H_1000', 'H_1700',
]
