# plotting.py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

LF_COLOR = '#e74c3c'
LF_LIGHT = '#f3c1b7'
LF_MID   = '#e99586'
HF_COLOR = '#1f77b4'
HF_LIGHT = '#c5d9ef'
HF_MID   = '#86b4df'
PLANE_COLOR = '#c7cfd6'

def create_phi_fig(time, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz, material):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=phi, mode='lines', name='Численное решение', line=dict(color='red')))
    if phi_fit is not None:
        fig.add_trace(go.Scatter(x=time, y=phi_fit, mode='lines', name='Аппроксимация', line=dict(color='blue', dash='dash')))
    fig.update_layout(
        title="Динамика угла φ",
        xaxis_title="Время (нс)",
        yaxis_title="Отклонение угла φ (градусы)",
        template="plotly_white",
        annotations=[{
            "x": 0.95,
            "y": 1.05,
            "text": (
                f"<b>Материал</b> {'FeFe' if material == '1' else 'GdFe'}<br>"
                f"<b>Магнитное поле</b> H = {H} Э<br>"
                f"<b>Температура</b> T = {T} K<br>"
                f"<b>HF</b> Аппроксимация: {approx_freqs_GHz[0]} ГГц; Аналитика: {theor_freqs_GHz[0]} ГГц<br>"
                f"<b>LF</b> Аппроксимация: {approx_freqs_GHz[1]} ГГц; Аналитика: {theor_freqs_GHz[1]} ГГц"
            ),
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "bordercolor": "black",
            "borderwidth": 1,
            "borderpad": 10,
            "bgcolor": "white"
        }],
        font=dict(size=18)
    )
    return fig

def create_theta_fig(time, theta, theta_fit):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=theta, mode='lines', name='Численное решение', line=dict(color='red')))
    if theta_fit is not None:
        fig.add_trace(go.Scatter(x=time, y=theta_fit, mode='lines', name='Аппроксимация', line=dict(color='blue', dash='dash')))
    fig.update_layout(
        title="Динамика угла θ",
        xaxis_title="Время (нс)",
        yaxis_title="Отклонение угла θ (градусы)",
        template="plotly_white",
        font=dict(size=18)
    )
    return fig

import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

def create_yz_fig(y, z, time, H_oe, colorscale="Plasma", n_bins=200):
    """
    y, z  : координаты (в ваших текущих 'сотых' единицах, где 0.01 = 1)
    time  : массив времени (любой шкалы)
    H_oe  : магнитное поле в Oe (в заголовке будет kOe)
    colorscale : строка Plotly colorscale (например 'Viridis', 'Cividis', 'Plasma', ...)
    n_bins : сколько цветовых "ступеней" для линии (больше -> плавнее, но тяжелее)
    """

    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = np.asarray(time, dtype=float)

    # (3) Замена единиц: 0.01 -> 1  => умножаем на 100
    y = 100.0 * y
    z = 100.0 * z

    # Одинаковые пределы + одинаковый масштаб (2)
    lim = float(np.max([np.max(np.abs(y)), np.max(np.abs(z))]))
    limits = (-1.1 * lim, 1.1 * lim)

    # Заголовок (как у T_fix)
    H_kOe = float(H_oe) / 1000.0
    title_text = f"H = {H_kOe:g} kOe"

    title_font = dict(family="Times New Roman, Times, serif", size=28, color="black")
    tick_font  = dict(family="Times New Roman, Times, serif", size=24, color="black")

    fig = go.Figure()

    # (1) Градиентная непрерывная линия:
    # Plotly не умеет "настоящий" градиент на одной линии, поэтому делаем линии-отрезки,
    # сгруппированные по бинам времени => немного трасс, визуально линия непрерывная.
    if len(y) >= 2:
        tmin, tmax = float(np.min(t)), float(np.max(t))
        if np.isclose(tmax, tmin):
            t_norm = np.zeros_like(t)
        else:
            t_norm = (t - tmin) / (tmax - tmin)

        # бин по "времени" для каждого сегмента (i -> i+1)
        seg_bin = np.floor(t_norm[:-1] * (n_bins - 1)).astype(int)
        seg_bin = np.clip(seg_bin, 0, n_bins - 1)

        # заранее выбираем цвета для бинов
        # sample_colorscale принимает позиции 0..1
        bin_pos = np.linspace(0, 1, n_bins)
        bin_colors = pc.sample_colorscale(colorscale, bin_pos)

        # для каждого бина собираем разорванный набор сегментов (через None)
        for b in range(n_bins):
            idx = np.where(seg_bin == b)[0]
            if idx.size == 0:
                continue
            xs, ys = [], []
            for i in idx:
                xs += [y[i], y[i+1], None]
                ys += [z[i], z[i+1], None]

            fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode="lines",
                line=dict(width=4, color=bin_colors[b]),
                hoverinfo="skip",
                showlegend=False
            ))

        # Colorbar отдельной "пустой" трассой (без точек на графике)
        fig.add_trace(go.Scatter(
            x=[limits[0]-10], y=[limits[0]-10],  # уводим за пределы
            mode="markers",
            marker=dict(
                size=0.1,
                color=[tmin, tmax],
                cmin=tmin, cmax=tmax,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Time (ns)", font=title_font),
                    tickfont=tick_font,
                    thickness=22,
                    len=0.90,
                    y=0.5,
                    yanchor="middle",
                    outlinewidth=1,
                    outlinecolor="black",
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
            opacity=0.0
        ))

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman, Times, serif", size=14, color="black"),
        margin=dict(l=90, r=120, t=10, b=70),
        title=dict(
            text=title_text,
            x=0.5, y=0.98,
            xref="paper", yref="paper",
            xanchor="center", yanchor="top",
            font=title_font
        ),
        xaxis=dict(
            title=dict(text="y (arb.units)", font=title_font, standoff=16),
            tickfont=tick_font,
            tickcolor="black",
            tickmode="linear",
            dtick=1,
            showline=True, linewidth=1, linecolor="black",
            mirror=True,
            showgrid=True, gridcolor="#cccccc", gridwidth=1,
            range=limits,
            tickangle=0,
        ),
        yaxis=dict(
            title=dict(text="z (arb.units)", font=title_font, standoff=16),
            tickfont=tick_font,
            tickcolor="black",
            tickmode="linear",
            dtick=1,
            showline=True, linewidth=1, linecolor="black",
            mirror=True,
            showgrid=True, gridcolor="#cccccc", gridwidth=1,
            range=limits,
            tickangle=0,
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=False
    )

    return fig


dot_size = 8
# def create_H_fix_fig(T_vals, H_fix_res, H, data=None):
#     (f1, t1), (f2, t2) = H_fix_res
#     fig = make_subplots(
#         rows=2, cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.1,
#     )
#     fig.add_trace(go.Scatter(x=T_vals, y=f1, mode='lines', name='HF', line=dict(color='blue')), row=1, col=1)
#     fig.add_trace(go.Scatter(x=T_vals, y=f2, mode='lines', name='LF', line=dict(color='red')), row=1, col=1)
#     if data is not None:
#         x_m, lf_freq, hf_freq, lf_tau, hf_tau = data
#         fig.add_trace(go.Scatter(x=x_m, y=hf_freq, mode='markers', name='HF (эксп.)', marker=dict(color='blue', size=dot_size)), row=1, col=1)
#         fig.add_trace(go.Scatter(x=x_m, y=lf_freq, mode='markers', name='LF (эксп.)', marker=dict(color='red', size=dot_size)), row=1, col=1)
#         fig.add_trace(go.Scatter(x=x_m, y=hf_tau, mode='markers', name='HF', marker=dict(color='blue', size=dot_size)), row=2, col=1)
#         fig.add_trace(go.Scatter(x=x_m, y=lf_tau, mode='markers', name='LF', marker=dict(color='red', size=dot_size)), row=2, col=1)
#     fig.add_trace(go.Scatter(x=T_vals, y=t1, mode='lines', name='HF', line=dict(color='blue')), row=2, col=1)
#     fig.add_trace(go.Scatter(x=T_vals, y=t2, mode='lines', name='LF', line=dict(color='red')), row=2, col=1)
#     fig.update_layout(
#         title={
#             'text': f"H = {H} Э",
#             'x': 0.5,
#             'xanchor': 'center',
#             'yanchor': 'top'
#         },
#         font=dict(size=18),
#         template="plotly_white",
#         showlegend=False
#     )
#     fig.update_yaxes(title_text="Частота (ГГц)", row=1, col=1)
#     fig.update_yaxes(title_text="Время затухания (нс)", row=2, col=1)
#     fig.update_xaxes(title_text="Температура (K)", row=2, col=1)
#     return fig
    
def create_H_fix_fig(T_vals, H_fix_res, H, data=None):
    T_plane = 333.0

    T_vals = np.asarray(T_vals, dtype=float)
    (f1, t1), (f2, t2) = H_fix_res
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)

    # сортировка по T (важно для линий)
    order = np.argsort(T_vals)
    T_vals = T_vals[order]
    f1 = f1[order]
    f2 = f2[order]

    # маски вокруг плоскости (оставляем точку на шве в обеих частях)
    mask_lo = T_vals <= T_plane
    mask_hi = T_vals >= T_plane

    T_lo, T_hi = T_vals[mask_lo], T_vals[mask_hi]
    f1_lo, f1_hi = f1[mask_lo], f1[mask_hi]
    f2_lo, f2_hi = f2[mask_lo], f2[mask_hi]

    title_font = dict(family="Times New Roman, Times, serif", size=28, color="black")
    tick_font  = dict(family="Times New Roman, Times, serif", size=24, color="black")

    fig = go.Figure()

    # Вертикальная линия T=333 (без прозрачности)
    fig.add_vline(
        x=T_plane,
        line_width=2,
        line_dash="dash",
        line_color="#7f7f7f"
    )

    # До плоскости: HF=f1, LF=f2
    fig.add_trace(go.Scatter(
        x=T_lo, y=f1_lo, mode='lines', name='HF',
        line=dict(width=2, color=HF_COLOR)
    ))
    fig.add_trace(go.Scatter(
        x=T_lo, y=f2_lo, mode='lines', name='LF',
        line=dict(width=2, color=LF_COLOR)
    ))

    # После плоскости: МЕНЯЕМ МЕСТАМИ (HF=f2, LF=f1)
    fig.add_trace(go.Scatter(
        x=T_hi, y=f2_hi, mode='lines', name='HF',
        line=dict(width=2, color=HF_COLOR)
    ))
    fig.add_trace(go.Scatter(
        x=T_hi, y=f1_hi, mode='lines', name='LF',
        line=dict(width=2, color=LF_COLOR)
    ))

    if data is not None:
        T_exp  = np.asarray(data[0], dtype=float)
        lf_exp = np.asarray(data[1], dtype=float)
        hf_exp = np.asarray(data[2], dtype=float)
    
        # --- 1) базовый swap эксп. по плоскости T_plane (это остаётся) ---
        y_lf = lf_exp.copy()
        y_hf = hf_exp.copy()
        m_plane = (T_exp > T_plane)
        tmp = y_lf[m_plane].copy()
        y_lf[m_plane] = y_hf[m_plane]
        y_hf[m_plane] = tmp
    
        # --- 2) пересечения теории по смене знака diff_th = f1 - f2 ---
        diff_th = np.asarray(f1, dtype=float) - np.asarray(f2, dtype=float)
    
        # (опционально, чтобы не ловить шум около нуля)
        eps0 = 0.0  # можешь поставить, например, 1e-12
        d = diff_th.copy()
        d[np.abs(d) <= eps0] = 0.0
    
        cross_T = []
    
        # точные нули (если вдруг есть)
        zero_idx = np.where(d == 0.0)[0]
        for i in zero_idx:
            cross_T.append(float(T_vals[i]))
    
        # смены знака между соседними узлами + линейная интерполяция
        for i in range(len(T_vals) - 1):
            d0, d1 = d[i], d[i + 1]
            if d0 == 0.0 or d1 == 0.0:
                continue
            if d0 * d1 < 0.0:
                T0, T1 = float(T_vals[i]), float(T_vals[i + 1])
                # d(T) линейна на отрезке -> T_cross = T0 - d0*(T1-T0)/(d1-d0)
                T_cross = T0 - d0 * (T1 - T0) / (d1 - d0)
                cross_T.append(float(T_cross))
    
        # убираем дубликаты (на случай нулей + интерполяции рядом)
        if cross_T:
            cross_T = np.array(sorted(cross_T), dtype=float)
            cross_T = cross_T[np.concatenate([[True], np.diff(cross_T) > 1e-9])].tolist()
    
        # --- 3) дополнительный swap эксп. по пересечениям ---
        if len(cross_T) >= 2:
            T_low, T_high = float(cross_T[0]), float(cross_T[-1])
            m_cross = (T_exp >= T_low) & (T_exp <= T_high)
        elif len(cross_T) == 1:
            T1 = float(cross_T[0])
            m_cross = (T_exp >= T1)
        else:
            m_cross = np.zeros_like(T_exp, dtype=bool)
    
        tmp = y_lf[m_cross].copy()
        y_lf[m_cross] = y_hf[m_cross]
        y_hf[m_cross] = tmp
    
        # --- 4) рисуем экспериментальные точки (после обоих swap) ---
        fig.add_trace(go.Scatter(
            x=T_exp, y=y_lf, mode='markers', name='LF (эксп.)',
            marker=dict(color=LF_COLOR, size=dot_size, line=dict(width=1, color="#000000"))
        ))
        fig.add_trace(go.Scatter(
            x=T_exp, y=y_hf, mode='markers', name='HF (эксп.)',
            marker=dict(color=HF_COLOR, size=dot_size, line=dict(width=1, color="#000000"))
        ))

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman, Times, serif", size=14, color="black"),
        margin=dict(l=90, r=20, t=10, b=70),
        title=dict(
            text=f"H = {H/1000} kOe",
            x=0.5, y=0.98,
            xref='paper',
            yref='paper',
            xanchor='center', yanchor='top',
            font=title_font
        ),
        xaxis=dict(
            title=dict(text="Temperature (K)", font=title_font, standoff=16),
            tickfont=tick_font,
            tickcolor="black",
            tickangle=0,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="#cccccc",
            gridwidth=1,
        ),
        yaxis=dict(
            title=dict(text="Frequency (GHz)", font=title_font, standoff=16),
            tickfont=tick_font,
            tickcolor="black",
            tickangle=0,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="#cccccc",
            gridwidth=1,
        ),
        showlegend=False
    )

    fig.add_annotation(
        x=T_plane + 1,
        y=0.5,
        xref="x",
        yref="paper",
        text=r"T<sub>M</sub>",
        showarrow=False,
        font=dict(family="Times New Roman, Times, serif", size=28, color="black"),
        xanchor="left",
        yanchor="middle"
    )

    return fig

# def create_T_fix_fig(H_vals, T_fix_res, T, data=None):
#     (f1, t1), (f2, t2) = T_fix_res
#     fig = make_subplots(
#         rows=2, cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.1,
#     )
#     fig.add_trace(go.Scatter(x=H_vals, y=f1, mode='lines', name='HF', line=dict(color='blue')), row=1, col=1)
#     fig.add_trace(go.Scatter(x=H_vals, y=f2, mode='lines', name='LF', line=dict(color='red')), row=1, col=1)
#     if data is not None:
#         x_m, lf_freq, hf_freq, lf_tau, hf_tau = data
#         fig.add_trace(go.Scatter(x=x_m, y=hf_freq, mode='markers', name='HF (эксп.)', marker=dict(color='blue', size=dot_size)), row=1, col=1)
#         fig.add_trace(go.Scatter(x=x_m, y=lf_freq, mode='markers', name='LF (эксп.)', marker=dict(color='red', size=dot_size)), row=1, col=1)
#         fig.add_trace(go.Scatter(x=x_m, y=hf_tau, mode='markers', name='HF', marker=dict(color='blue', size=dot_size)), row=2, col=1)
#         fig.add_trace(go.Scatter(x=x_m, y=lf_tau, mode='markers', name='LF', marker=dict(color='red', size=dot_size)), row=2, col=1)
#     fig.add_trace(go.Scatter(x=H_vals, y=t1, mode='lines', name='HF', line=dict(color='blue')), row=2, col=1)
#     fig.add_trace(go.Scatter(x=H_vals, y=t2, mode='lines', name='LF', line=dict(color='red')), row=2, col=1)
#     fig.update_layout(
#         title={
#             'text': f"T = {T} K",
#             'x': 0.5,
#             'xanchor': 'center',
#             'yanchor': 'top'
#         },
#         font=dict(size=18),
#         template="plotly_white",
#         showlegend=False
#     )
#     fig.update_yaxes(title_text="Частота (ГГц)", row=1, col=1)
#     fig.update_yaxes(title_text="Время затухания (нс)", row=2, col=1)
#     fig.update_xaxes(title_text="Магнитное поле (Э)", row=2, col=1)
#     return fig

def create_T_fix_fig(H_vals, T_fix_res, T, data=None):
    H_kOe = np.asarray(H_vals, dtype=float) / 1000.0
    (f1, t1), (f2, t2) = T_fix_res
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)

    # на всякий случай сортировка по H
    order = np.argsort(H_kOe)
    H_kOe = H_kOe[order]
    f1 = f1[order]
    f2 = f2[order]

    title_font = dict(family="Times New Roman, Times, serif", size=28, color="black")
    tick_font  = dict(family="Times New Roman, Times, serif", size=24, color="black")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=H_kOe, y=f1, mode='lines', name='HF',
        line=dict(width=2, color=HF_COLOR)
    ))
    fig.add_trace(go.Scatter(
        x=H_kOe, y=f2, mode='lines', name='LF',
        line=dict(width=2, color=LF_COLOR)
    ))

    if data is not None:
        fig.add_trace(go.Scatter(
            x=np.asarray(data[0], dtype=float)/1000.0, y=np.asarray(data[1], dtype=float),
            mode='markers', name='LF (эксп.)',
            marker=dict(color=LF_COLOR, size=dot_size, line=dict(width=1, color="#000000"))
        ))
        fig.add_trace(go.Scatter(
            x=np.asarray(data[0], dtype=float)/1000.0, y=np.asarray(data[2], dtype=float),
            mode='markers', name='HF (эксп.)',
            marker=dict(color=HF_COLOR, size=dot_size, line=dict(width=1, color="#000000"))
        ))

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman, Times, serif", size=14, color="black"),
        margin=dict(l=90, r=20, t=10, b=70),
        title=dict(
            text=f"T = {T} K",
            x=0.5, y=0.98,
            xref='paper',
            yref='paper',
            xanchor='center', yanchor='top',
            font=title_font
        ),
        xaxis=dict(
            title=dict(text="Magnetic field (kOe)", font=title_font, standoff=16),
            tickfont=tick_font,
            tickcolor="black",
            tickangle=0,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="#cccccc",
            gridwidth=1,
        ),
        yaxis=dict(
            title=dict(text="Frequency (GHz)", font=title_font, standoff=16),
            tickfont=tick_font,
            tickcolor="black",
            tickangle=0,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="#cccccc",
            gridwidth=1,
        ),
        showlegend=False
    )

    return fig

def create_freq_fig(T_vals, H_vals, freq_res_grid):
    T_plane = 333.0
    
    T_vals = np.asarray(T_vals, dtype=float)
    H_kOe  = np.asarray(H_vals, dtype=float) / 1000.0
    
    (f1_grid, _), (f2_grid, _) = freq_res_grid
    
    mask_lo = T_vals <= T_plane
    mask_hi = T_vals >= T_plane
    T_lo = T_vals[mask_lo]
    T_hi = T_vals[mask_hi]
    f1_lo = f1_grid[mask_lo, :]
    f1_hi = f1_grid[mask_hi, :]
    f2_lo = f2_grid[mask_lo, :]
    f2_hi = f2_grid[mask_hi, :]

    zmin = float(np.nanmin([np.nanmin(f1_grid), np.nanmin(f2_grid)]))
    zmax = float(np.nanmax([np.nanmax(f1_grid), np.nanmax(f2_grid)]))

    Hmin, Hmax = float(np.nanmin(H_kOe)), float(np.nanmax(H_kOe))
    x_plane = np.array([[Hmin, Hmax],
                        [Hmin, Hmax]])
    y_plane = np.array([[T_plane, T_plane],
                        [T_plane, T_plane]])
    z_plane = np.array([[zmin, zmin],
                        [zmax, zmax]])

    title_font = dict(family="Times New Roman, Times, serif", size=18, color="black")
    tick_font  = dict(family="Times New Roman, Times, serif", size=14, color="black")

    fig = go.Figure()

    # полупрозрачная плоскость T = 333 K
    fig.add_trace(go.Surface(
        x=x_plane, y=y_plane, z=z_plane,
        colorscale=[[0, PLANE_COLOR], [1, PLANE_COLOR]],
        showscale=False,
        opacity=0.25,
        name=f"T = {T_plane} K",
        hoverinfo="skip"
    ))

    title_pad_lines = 10
    pad = f"<br>" * int(title_pad_lines)

    fig.add_trace(go.Surface(
        z=f1_lo, x=H_kOe, y=T_lo,
        colorscale=[[0.0, HF_LIGHT], [0.45, HF_MID], [1.0, HF_COLOR]],
        cmin=f1_grid.min(), cmax=f1_grid.max(),
        showscale=False, name='HF'
    ))
    fig.add_trace(go.Surface(
        z=f2_hi, x=H_kOe, y=T_hi,
        colorscale=[[0.0, HF_LIGHT], [0.45, HF_MID], [1.0, HF_COLOR]],
        cmin=f2_grid.min(), cmax=f2_grid.max(),
        showscale=False, name='HF'
    ))
    fig.add_trace(go.Surface(
        z=f2_lo, x=H_kOe, y=T_lo,
        colorscale=[[0.0, LF_LIGHT], [0.45, LF_MID], [1.0, LF_COLOR]],
        cmin=f2_grid.min(), cmax=f2_grid.max(),
        showscale=False, name='LF'
    ))
    fig.add_trace(go.Surface(
        z=f1_hi, x=H_kOe, y=T_hi,
        colorscale=[[0.0, LF_LIGHT], [0.45, LF_MID], [1.0, LF_COLOR]],
        cmin=f1_grid.min(), cmax=f1_grid.max(),
        showscale=False, name='LF'
    ))

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman, Times, serif", size=14, color="black"),
        margin=dict(l=120, r=20, t=10, b=70),
        scene=dict(
            xaxis=dict(
                title=dict(text=pad + f"Magnetic field (kOe)", font=title_font),
                tickfont=tick_font,
                tickcolor="black",
            ),
            yaxis=dict(
                title=dict(text=pad + f"Temperature (K)", font=title_font),
                tickfont=tick_font,
                tickcolor="black",
            ),
            zaxis=dict(
                title=dict(text=pad + f"Frequency (GHz)", font=title_font),
                tickfont=tick_font,
                tickcolor="black",
            ),
            camera=dict(projection=dict(type='orthographic')),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3),
        ),
        legend=dict(
            font=dict(family="Times New Roman, Times, serif", size=12, color="black"),
            bgcolor="rgba(255,255,255,0.7)"
        )
    )

    return fig

# def create_freq_fig(T_vals, H_vals, freq_array1, freq_array2):
#     fig = go.Figure(
#         data=[
#             go.Surface(z=freq_array1, x=H_vals, y=T_vals,
#                        colorscale=[[0, 'rgb(255, 182, 193)'], [1, 'rgb(255, 0, 0)']],
#                        showscale=False, name='HF'),
#             go.Surface(z=freq_array2, x=H_vals, y=T_vals,
#                        colorscale=[[0, 'rgb(173, 216, 230)'], [1, 'rgb(0, 0, 255)']],
#                        showscale=False, name='LF')
#         ],
#         layout=go.Layout(
#             title="Частоты LF и HF в зависимости от H и T",
#             scene=dict(
#                 xaxis_title='Магнитное поле (Э)',
#                 yaxis_title='Температура (K)',
#                 zaxis_title='Частота (ГГц)'
#             ),
#             font=dict(size=14),
#             template="plotly_white"
#         )
#     )
#     return fig

def create_phase_fig(T_vals, theta_0):
    H_kOe = np.arange(0, 4001, 50) / 1000
    theta_0 = theta_0.T
    custom_colorscale = [
        [0.00, 'rgb(0, 0, 0)'],        # black
        [0.31, 'rgb(0, 0, 255)'],      # blue
        [0.62, 'rgb(0, 128, 0)'],      # green
        [0.93, 'rgb(255, 255, 0)'],    # yellow
        [1.00, 'rgb(255, 255, 255)']   # white
    ]
    heat = go.Heatmap(
        x=T_vals,
        y=H_kOe,
        z=theta_0,
        colorscale=custom_colorscale,
        colorbar=dict(
            title=dict(
                text=r'θ₀ (rad)',
                side='right'
            ),
            tickmode='array',
            tickvals=[0, np.pi / 4, np.pi / 2],
            ticktext=['0', 'π/4', 'π/2'],
            outlinewidth=1
        ),
        zmin=0, zmax=np.pi / 2,
        showscale=True
    )

    fig = go.Figure(data=[heat])

    contour = go.Contour(
        x=T_vals,
        y=H_kOe,
        z=theta_0,
        showscale=False,
        contours=dict(
            start=0.01, end=0.01, size=0.01,
            coloring='none'
        ),
        line=dict(width=1.5, color='white')
    )
    fig.add_trace(contour)

    mask = (theta_0[:, 0] > 0) & (theta_0[:, 0] < 0.1)
    if np.any(mask):
        idx_min, idx_max = np.where(mask)[0][[0, -1]]
        y_noncol = H_kOe[idx_max] + 0.3
        y_col    = H_kOe[idx_min] - 0.3
        fig.add_annotation(x=T_vals[0], y=y_noncol,
                           text='non‑collinear',
                           showarrow=False, font=dict(color='white', size=14),
                           xanchor='left', yanchor='bottom')
        fig.add_annotation(x=T_vals[0], y=y_col,
                           text='collinear',
                           showarrow=False, font=dict(color='white', size=14),
                           xanchor='left', yanchor='top')

    fig.update_layout(
        xaxis=dict(title='T (K)', tickcolor="black", range=[T_vals.min(), T_vals.max()]),
        yaxis=dict(title='H (kOe)', tickcolor="black", range=[H_kOe.min(), H_kOe.max()]),
        template='plotly_white',
        margin=dict(l=60, r=40, t=40, b=60)
    )

    return fig

    
__all__ = [
    'create_phi_fig', 'create_theta_fig', 'create_yz_fig', 'create_H_fix_fig', 'create_phase_fig',
    'create_T_fix_fig', 'create_freq_fig',
]
