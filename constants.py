# constants.py
import numpy as np
import pandas as pd
from typing import Iterable
from pathlib import Path

def load_exp(path: Path) -> np.ndarray:
    """
    Читает Excel-файл с данными по одному срезу (H = const или T = const)
    и возвращает numpy-массив формы (9, N):

    [
        axis_values,   # меняющаяся ось: H или T
        f_lf,
        f_hf,
        tau_lf,
        tau_hf,
        err_f_lf,
        err_f_hf,
        err_tau_lf,
        err_tau_hf
    ]

    Если какой-то строки нет, возвращается массив NaN той же длины.
    """

    df = pd.read_excel(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{path.name}: слишком мало столбцов для данных.")

    def _parse_filename(name: str) -> tuple[str, float]:
        stem = Path(name).stem
        if stem.upper().startswith("T_"):
            return "T", float(stem.split("_", 1)[1])
        if stem.upper().startswith("H_"):
            return "H", float(stem.split("_", 1)[1])
        raise ValueError(f"{name}: имя файла должно начинаться с T_<val> или H_<val>.")

    def _to_float_array(series) -> np.ndarray:
        return np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
  
    fixed_type, fixed_value = _parse_filename(path.name)
    axis_values = _to_float_array(df.iloc[0, 1:])
    if np.any(~np.isfinite(axis_values)):
        raise ValueError(f"{path.name}: не удалось прочитать значения оси (первая строка).")
    if fixed_type == "T":
        axis_values = axis_values * 10

    def _extract_rows(df_body: pd.DataFrame) -> dict[str, np.ndarray]:
        """Извлекает числовые ряды из таблицы по текстовым меткам."""
        def _normalize_label(label: str) -> str:
            normalized = label.strip().lower()
            normalized = normalized.replace("ё", "е")
            for ch in [" ", ".", ",", "\t"]:
                normalized = normalized.replace(ch, "")
            return normalized
  
        label_map = {
            "частотанчггц": "f_lf",
            "частотавчггц": "f_hf",
            "времязатуханиянчнс": "tau_lf",
            "времязатуханиявчнс": "tau_hf",
            "погрчастотанчггц": "err_f_lf",
            "погрчастотавчггц": "err_f_hf",
            "погрвремязатуханиянчнс": "err_tau_lf",
            "погрвремязатуханиявчнс": "err_tau_hf",
        }
        result: dict[str, np.ndarray] = {}
        for _, row in df_body.iterrows():
            label_raw = str(row.iloc[0])
            key = _normalize_label(label_raw)
            mapped = label_map.get(key)
            if mapped is None:
                continue
            result[mapped] = _to_float_array(row.iloc[1:])
        return result
  
    rows_map = _extract_rows(df.iloc[1:, :])

    if rows_map.get("f_lf") is None and rows_map.get("f_hf") is None:
        raise ValueError(f"{path.name}: не найдено строк с частотами.")

    n = len(axis_values)

    def get_row(key: str) -> np.ndarray:
        row = rows_map.get(key)
        if row is None:
            return np.full(n, np.nan, dtype=float)

        arr = np.asarray(row, dtype=float)

        if arr.size != n:
            raise ValueError(
                f"{path.name}: строка '{key}' имеет длину {arr.size}, ожидалось {n}."
            )

        return arr

    f_lf = get_row("f_lf")
    f_hf = get_row("f_hf")
    tau_lf = get_row("tau_lf")
    tau_hf = get_row("tau_hf")

    err_f_lf = get_row("err_f_lf")
    err_f_hf = get_row("err_f_hf")
    err_tau_lf = get_row("err_tau_lf")
    err_tau_hf = get_row("err_tau_hf")

    result = np.vstack([
        axis_values,
        f_lf,
        f_hf,
        tau_lf,
        tau_hf,
        err_f_lf,
        err_f_hf,
        err_tau_lf,
        err_tau_hf,
    ])
    return result

# Данные
# T_293 = np.array([[1000, 1200, 1400, 1600, 1800, 2000],
#                   [9.17440137,  9.423370201,  9.735918686,  10.01455683,  10.37994595,  10.5903492],
#                   [29.35937721, 31.65155559,  30.2486405,   30.17815415,  29.94237192,  27.36357678],
#                   [0.2326714,   0.2440502,    0.2528819,    0.2692589,	  0.2764174,    0.3013753],
#                   [0.01385995,  0.01490037,   0.01370499,   0.01470922,   0.0150656,    0.02085652]])
# T_298 = np.array([[300,	400,	500,	600,	700,	800,	900,	1000,	1100,	1200,	1300,	1400,	1500,	1600,	1700],
#                   [7.917725597,	8.076488856,	8.308853218,	8.511097697,	8.768619546,	8.986239461,	9.193628098,	9.266547553,	9.529545031,	9.797292885,	9.989582687,	10.2246736,	  10.45394883,	10.64744636,	10.86148998],
#                   [35.74602832,	35.69551319,	36.27748124,	35.91225848,	36.11820021,	36.15389806,	35.05317194,	34.51988763,	33.93198569,	33.95446619,	34.26943193,	33.58995775,	33.33629316,	33.36014522,	33.66639751],
#                   [0.386641778,	0.48149481,	  0.383780452,	0.372675986,	0.437156799,	0.426795871,	0.383005992,	0.41417349,	  0.42977848,	  0.431501565,	0.489608943,	0.431084878,	0.445555755,	0.430411838,	0.490947342],
#                   [0.033426825,	0.031589732,	0.032365894,	0.030020148,	0.033882985,	0.031254696,	0.033037456,	0.026293099,	0.028122181,	0.025981535,	0.025182351,	0.024667935,	0.024979283,	0.023982741,	0.027571491]])
# T_308 =  np.array([[300,	400,	500,	600,	700,	800,	900,	1000,	1100,	1200,	1300,	1400,	1500,	1600,	1700],
#                    [8.937549551,	9.183303801,	9.436465239,	9.672472021,	9.799494481,	9.888207347,	10.0839591,	  10.22131939,	10.3813667,	  10.57739003,	10.7075884,	  10.90387457,	11.14477707,	11.28980085,	11.55331547],
#                    [22.7019311,	  22.67961303,	22.00522988,	21.67647155,	22.41350342,	22.24309327,	21.6154545,	  21.49189617,	21.55743187,	21.23982667,	21.04952922,	20.04489697,	19.10347648,	20.05336376,	18.65852376],
#                    [0.189330769,	0.205763739,	0.209861742,	0.186501848,	0.23591113,	  0.243050325,	0.240669259,	0.240415501,	0.268157207,	0.279112256,	0.292817478,	0.342335456,	0.291896044,	0.284999857,	0.283908835],
#                    [0.029700689,	0.027460218,	0.02700383,	  0.026121174,	0.022535344,	0.02116828,	  0.022816963,	0.024243045,	0.02556581,	  0.021633368,	0.023212102,	0.017383083,	0.033965081,	0.020187841,	0.031583577]])
# T_310 = np.array([[1000, 1200, 1400, 1600, 1800],
#                   [9.608579,    10.1564,	    10.48156,     10.75176,	    10.5243],
#                   [19.53544,    20.56842378,  19.6038,      18.23266,	    22.76495312],
#                   [0.1791479,   0.2147599,    0.1493049,    0.1797168,    0.2107221],
#                   [0.025845326, 0.02878221,   0.05083024,   0.03752954,   None]])
# T_320 = np.array([[300,	400,	500,	600,	700,	800,	900,	1000,	1100,	1200,	1300,	1400,	1500,	1600,	1700],
#                   [10.72802486,	11.0638429,	  11.57000741,	10.56978123,	11.93735544,	12.48866159,	11.16335263,	10.3322006,	  10.55031077,	9.933082495,	9.557305257,	8.978562914,	8.483450777,	8.09590908,	  7.670299162],
#                   [16.3200493,	16.84256107,	14.82584226,	11.6747025,	  13.10077717,	14.24688068,	12.01294889,	11.95257493,	11.9264299,	  11.97678067,	12.2944839,	  12.36919772,	12.62161187,	12.78916765,	12.72348708],
#                   [0.090305018,	0.087247847,	0.08579766,	  0.050002629,	0.044183669,	0.07908596,	  0.092696218,	0.050000068,	0.032577073,	0.030840242,	0.055008561,	0.043560667,	0.072294673,	0.086216721,	0.088999581],
#                   [0.08534056, 	0.00504612,	  0.02441555,	  0.148976329,	0.079582647,	0.058288813,	0.263201365,	0.16604669,	  0.139542414,	0.144755406,	0.15606704,	  0.154683336, 	0.164758408, 	0.211400921, 	0.188879884]])
# T_323 = np.array([[1000, 1200, 1400, 1600, 1800],
#                   [8.0366,      6.103743,     1.262356544,  3.568812289,  3.787543515],
#                   [10.16,       10.49205139,  11.05093022,  10.84827952,  11.39313551],
#                   [0.1512544,   0.1455957,    2.86294,      0.8830667,    0.1959008],
#                   [0.1470505,   0.2757782,    0.1217176,    0.110712,     0.1518173]])
# H_1000 = np.array([[298,	303,	308,	313,	318,	323,	328,	333],
#                    [9.282562466,	9.680674837,	10.18339115,	10.83380689,	9.222901978,	9.617573296,	5.215978832,	4.250938123],
#                    [36.07808758,	25.77246081,	19.52522355,	14.93828927,	11.82878082,	11.07828849,	10.90561062,	9.629826483],
#                    [0.331578945,	0.452703028,	0.312870677,	0.250044992,	0.036318981,	0.036455964,	0.107359658,	0.103978657],
#                    [0.02789031, 	0.026407094,	0.035760979,	0.024823965,	0.142086126,	0.08607823,	  0.098408155,	0.026101735]])
# # H_1000[0] += 5
# H_1700 = np.array([[298,	303,	308,	313,	318,	323,	328,	333],
#                    [10.64395404,	11.08859404,	11.33419432,	11.76530759,	9.800271063,	8.937050334,	4.076657625,	2.653114246],
#                    [37.43165126,	29.22627049,	23.1846905,  	17.43323314,	11.94522728,	12.73961325,	11.40811893,	12.09133868],
#                    [0.818144961,	0.484518225,	0.350349669,	0.215105189,	0.072835581,	0.119587347,	0.050000003,	0.048582682],
#                    [0.030453149,	0.027546426,	0.020729218,	0.033482177,	0.295981413,	0.130111567,	0.107177504,	0.052455122]])
# # H_1700[0] += 5

# T_298 = np.load('T_298.npy')
# T_308 = np.load('T_308.npy')
# T_320 = np.load('T_320.npy')
# H_1000 = np.load('H_100.npy')
# H_1700 = np.load('H_170.npy')

T_298 = load_exp(Path('measurement/T_298.xlsx'))
T_308 = load_exp(Path('measurement/T_308.xlsx'))
T_320 = load_exp(Path('measurement/T_320.xlsx'))
H_1000 = load_exp(Path('measurement/H_1000.xlsx'))
H_1700 = load_exp(Path('measurement/H_1700.xlsx'))

# Исходные параметры (Материал 1)
H_step = 10
H_lim = 2000
H_vals = np.arange(0, H_lim + 1, H_step)
T_vals_1 = np.linspace(290, 350, 601)
T_vals_2 = np.linspace(290, 350, 61)
T_init = 325.1

gamma = 1.76e7              # рад/(с·Oe)
lam_1 = 12500
lam_2 = 10000 # Заглушка!
alpha_1 = 1e-3
alpha_2 = 1.7e-2
h_IFE = 7500                # Ое
delta_t = 250e-15           # с

def K_T(T: Iterable[float] | float) -> np.ndarray:
    """Анизотропия как функция температуры."""
    return 0.522 * (T - 358.0) ** 2

def chi_func(m: np.ndarray | float, M: np.ndarray | float, lam: float) -> np.ndarray | float:
    """Вычисление магнитной восприимчивости."""
    denom = 1.0 - (m**2) / (M**2 + 1e-16)
    denom = np.where(denom == 0, np.nan, denom)
    return 1.0 / (lam * denom)

# Загрузка данных для материала 1
m_array_1 = np.load('m_array_new.npy')
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
    H_vals = np.arange(0, 4001, 50)
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
    'T_298', 'T_308', 'T_320',
    'H_1000', 'H_1700',
]
