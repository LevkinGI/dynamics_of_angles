# plotting.py
import plotly.graph_objs as go
import plotly.colors as pc
import numpy as np

LF_COLOR = '#e74c3c'
LF_LIGHT = '#f3c1b7'
LF_MID   = '#e99586'
HF_COLOR = '#1f77b4'
HF_LIGHT = '#c5d9ef'
HF_MID   = '#86b4df'
PLANE_COLOR = '#c7cfd6'



def _normalize_language(language='eng'):
    if language is None:
        return 'eng'
    language = str(language).lower()
    if language not in {'eng', 'ru'}:
        raise ValueError("language must be either 'eng' or 'ru'")
    return language


def _tr(language='eng'):
    language = _normalize_language(language)
    if language == 'ru':
        return {
            'numerical_solution': 'Численное решение',
            'fit': 'Аппроксимация',
            'phi_title': 'Динамика угла φ',
            'theta_title': 'Динамика угла θ',
            'time_ns': 'Время (нс)',
            'phi_axis': 'Отклонение угла φ (градусы)',
            'theta_axis': 'Отклонение угла θ (градусы)',
            'material': 'Материал',
            'magnetic_field': 'Магнитное поле',
            'temperature': 'Температура',
            'approximation': 'Аппроксимация',
            'analytics': 'Аналитика',
            'theory': 'Теория',
            'time_colorbar': 'Время (нс)',
            'arb_units': 'отн. ед.',
            'temperature_axis': 'Температура (K)',
            'frequency_axis': 'Частота (ГГц)',
            'magnetic_field_axis': 'Магнитное поле (кЭ)',
            'experiment_suffix': ' (эксп.)',
            'phase_colorbar': 'θ₀ (рад)',
            'phase_x': 'T (K)',
            'phase_y': 'H (кЭ)',
            'field_title': 'Поле H = {value} кЭ',
            'temperature_title': 'Температура T = {value} K',
            'tm_label': r'T<sub>M</sub>',
        }
    return {
        'numerical_solution': 'Numerical solution',
        'fit': 'Fit',
        'phi_title': 'Dynamics of angle φ',
        'theta_title': 'Dynamics of angle θ',
        'time_ns': 'Time (ns)',
        'phi_axis': 'Angle deviation φ (degrees)',
        'theta_axis': 'Angle deviation θ (degrees)',
        'material': 'Material',
        'magnetic_field': 'Magnetic field',
        'temperature': 'Temperature',
        'approximation': 'Approximation',
        'analytics': 'Analytical',
        'theory': 'Theory',
        'time_colorbar': 'Time (ns)',
        'arb_units': 'arb. units',
        'temperature_axis': 'Temperature (K)',
        'frequency_axis': 'Frequency (GHz)',
        'magnetic_field_axis': 'Magnetic field (kOe)',
        'experiment_suffix': ' (exp.)',
        'phase_colorbar': 'θ₀ (rad)',
        'phase_x': 'T (K)',
        'phase_y': 'H (kOe)',
        'field_title': 'Magnetic field H = {value} kOe',
        'temperature_title': 'Temperature T = {value} K',
        'tm_label': r'T<sub>M</sub>',
    }

def create_phi_fig(time, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz, material, language='eng'):
    tr = _tr(language)
    hf_app = "—" if (approx_freqs_GHz[0] is None) else f"{approx_freqs_GHz[0]:.1f}"
    lf_app = "—" if (approx_freqs_GHz[1] is None) else f"{approx_freqs_GHz[1]:.1f}"
    hf_th  = "—" if (theor_freqs_GHz[0] is None) else f"{theor_freqs_GHz[0]:.1f}"
    lf_th  = "—" if (theor_freqs_GHz[1] is None) else f"{theor_freqs_GHz[1]:.1f}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=phi, mode='lines', name=tr['numerical_solution'], line=dict(color='red')))
    if phi_fit is not None:
        fig.add_trace(go.Scatter(x=time, y=phi_fit, mode='lines', name=tr['fit'], line=dict(color='blue', dash='dash')))
    fig.update_layout(
        title=tr['phi_title'],
        xaxis_title=tr['time_ns'],
        yaxis_title=tr['phi_axis'],
        template="plotly_white",
        annotations=[{
            "x": 0.95,
            "y": 1.05,
            "text": (
                f"<b>{tr['material']}</b> {'FeFe' if material == '1' else 'GdFe'}<br>"
                f"<b>{tr['magnetic_field']}</b> H = {H} Oe<br>"
                f"<b>{tr['temperature']}</b> T = {T} K<br>"
                f"<b>HF</b> {tr['approximation']}: {hf_app} GHz; {tr['analytics']}: {hf_th} GHz<br>"
                f"<b>LF</b> {tr['approximation']}: {lf_app} GHz; {tr['analytics']}: {lf_th} GHz"
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

def create_theta_fig(time, theta, theta_fit, language='eng'):
    tr = _tr(language)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=theta, mode='lines', name=tr['numerical_solution'], line=dict(color='red')))
    if theta_fit is not None:
        fig.add_trace(go.Scatter(x=time, y=theta_fit, mode='lines', name=tr['fit'], line=dict(color='blue', dash='dash')))
    fig.update_layout(
        title=tr['theta_title'],
        xaxis_title=tr['time_ns'],
        yaxis_title=tr['theta_axis'],
        template="plotly_white",
        font=dict(size=18)
    )
    return fig

def create_yz_fig(
    y, z, time, H_oe,
    colorscale="Plasma", n_bins=300,
    pulse2_on=False,
    pulse2_time=None,
    pulse2_axis_on=False,
    pulse2_dir_on=False,
    knock_scale=1.0,
    language='eng',
):
    tr = _tr(language)
    y = 100.0 * y
    z = 100.0 * z

    lim = float(np.max([np.max(np.abs(y)), np.max(np.abs(z))]))
    limits = (-1.1 * lim, 1.1 * lim)

    H_kOe = float(H_oe) / 1000.0
    title_text = tr['field_title'].format(value=f"{H_kOe:g}")

    title_font = dict(family="Times New Roman, Times, serif", size=28, color="black")
    tick_font  = dict(family="Times New Roman, Times, serif", size=24, color="black")

    fig = go.Figure()

    if len(y) >= 2:
        tmin, tmax = float(np.min(time)), float(np.max(time))
        if np.isclose(tmax, tmin):
            t_norm = np.zeros_like(time)
        else:
            t_norm = (time - tmin) / (tmax - tmin)

        seg_bin = np.floor(t_norm[:-1] * (n_bins - 1)).astype(int)
        seg_bin = np.clip(seg_bin, 0, n_bins - 1)

        bin_pos = np.linspace(0, 1, n_bins)
        bin_colors = pc.sample_colorscale(colorscale, bin_pos)

        for b in range(n_bins):
            idx = np.where(seg_bin == b)[0]
            if idx.size == 0:
                continue
            xs, ys = [], []
            for i in idx:
                xs += [y[i], y[i + 1], None]
                ys += [z[i], z[i + 1], None]

            fig.add_trace(go.Scattergl(
                x=xs, y=ys,
                mode="lines",
                line=dict(width=4, color=bin_colors[b]),
                hoverinfo="skip",
                showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=[limits[0] - 10], y=[limits[0] - 10],
            mode="markers",
            marker=dict(
                size=0.1,
                color=[tmin, tmax],
                cmin=tmin, cmax=tmax,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(text=tr['time_colorbar'], font=title_font),
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

        if pulse2_on and (pulse2_time is not None):
            idx = int(np.argmin(np.abs(time - pulse2_time)))
            y0 = float(y[idx])
            z0 = float(z[idx])

            sgn = -1.0 if pulse2_dir_on else 1.0
            dy, dz = (sgn * knock_scale, 0.0) if pulse2_axis_on else (0.0, sgn * knock_scale)

            fig.add_annotation(
                x=y0 + dy, y=z0 + dz,
                ax=y0, ay=z0,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=4,
                arrowcolor=LF_COLOR,
            )

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
            title=dict(text=f"y ({tr['arb_units']})", font=title_font, standoff=16),
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
            title=dict(text=f"z ({tr['arb_units']})", font=title_font, standoff=16),
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
    
def create_H_fix_fig(T_vals, H_fix_res, H, data=None, language='eng'):
    tr = _tr(language)
    T_plane = 333.0

    T_vals = np.asarray(T_vals, dtype=float)
    (f1, t1), (f2, t2) = H_fix_res
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)

    order = np.argsort(T_vals)
    T_vals = T_vals[order]
    f1 = f1[order]
    f2 = f2[order]

    mask_lo = T_vals <= T_plane
    mask_hi = T_vals >= T_plane

    T_lo, T_hi = T_vals[mask_lo], T_vals[mask_hi]
    f1_lo, f1_hi = f1[mask_lo], f1[mask_hi]
    f2_lo, f2_hi = f2[mask_lo], f2[mask_hi]

    title_font = dict(family="Times New Roman, Times, serif", size=28, color="black")
    tick_font  = dict(family="Times New Roman, Times, serif", size=24, color="black")

    fig = go.Figure()

    fig.add_vline(
        x=T_plane,
        line_width=2,
        line_dash="dash",
        line_color="#7f7f7f"
    )

    fig.add_trace(go.Scatter(
        x=T_lo, y=f1_lo, mode='lines', name='HF',
        line=dict(width=2, color=HF_COLOR)
    ))
    fig.add_trace(go.Scatter(
        x=T_lo, y=f2_lo, mode='lines', name='LF',
        line=dict(width=2, color=LF_COLOR)
    ))

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
        err_lf_exp = np.asarray(data[5], dtype=float)
        err_hf_exp = np.asarray(data[6], dtype=float)

        y_lf = lf_exp.copy()
        y_hf = hf_exp.copy()
        err_y_lf = err_lf_exp.copy()
        err_y_hf = err_hf_exp.copy()
        m_plane = T_exp > T_plane
        tmp = y_lf[m_plane].copy()
        y_lf[m_plane] = y_hf[m_plane]
        y_hf[m_plane] = tmp
        err_tmp = err_y_lf[m_plane].copy()
        err_y_lf[m_plane] = err_y_hf[m_plane]
        err_y_hf[m_plane] = err_tmp

        diff_th = np.asarray(f1, dtype=float) - np.asarray(f2, dtype=float)
        eps0 = 0.0
        d = diff_th.copy()
        d[np.abs(d) <= eps0] = 0.0

        cross_T = []
        zero_idx = np.where(d == 0.0)[0]
        for i in zero_idx:
            cross_T.append(float(T_vals[i]))

        for i in range(len(T_vals) - 1):
            d0, d1 = d[i], d[i + 1]
            if d0 == 0.0 or d1 == 0.0:
                continue
            if d0 * d1 < 0.0:
                T0, T1 = float(T_vals[i]), float(T_vals[i + 1])
                T_cross = T0 - d0 * (T1 - T0) / (d1 - d0)
                cross_T.append(float(T_cross))

        if cross_T:
            cross_T = np.array(sorted(cross_T), dtype=float)
            cross_T = cross_T[np.concatenate([[True], np.diff(cross_T) > 1e-9])].tolist()

        if len(cross_T) >= 2:
            T_low, T_high = float(cross_T[0]), float(cross_T[-1])
            m_cross = (T_exp >= T_low) & (T_exp <= T_high)
        elif len(cross_T) == 1:
            T1 = float(cross_T[0])
            m_cross = T_exp >= T1
        else:
            m_cross = np.zeros_like(T_exp, dtype=bool)

        tmp = y_lf[m_cross].copy()
        y_lf[m_cross] = y_hf[m_cross]
        y_hf[m_cross] = tmp
        err_tmp = err_y_lf[m_cross].copy()
        err_y_lf[m_cross] = err_y_hf[m_cross]
        err_y_hf[m_cross] = err_tmp

        fig.add_trace(go.Scatter(
            x=T_exp, y=y_lf, mode='markers', name='LF' + tr['experiment_suffix'],
            error_y=dict(type="data", array=err_y_lf),
            marker=dict(color=LF_COLOR, size=dot_size, line=dict(width=1, color="#000000"))
        ))
        fig.add_trace(go.Scatter(
            x=T_exp, y=y_hf, mode='markers', name='HF' + tr['experiment_suffix'],
            error_y=dict(type="data", array=err_y_hf),
            marker=dict(color=HF_COLOR, size=dot_size, line=dict(width=1, color="#000000"))
        ))

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman, Times, serif", size=14, color="black"),
        margin=dict(l=90, r=20, t=10, b=70),
        title=dict(
            text=tr['field_title'].format(value=f"{H/1000:g}"),
            x=0.5, y=0.98,
            xref='paper',
            yref='paper',
            xanchor='center', yanchor='top',
            font=title_font
        ),
        xaxis=dict(
            title=dict(text=tr['temperature_axis'], font=title_font, standoff=16),
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
            title=dict(text=tr['frequency_axis'], font=title_font, standoff=16),
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
        text=tr['tm_label'],
        showarrow=False,
        font=dict(family="Times New Roman, Times, serif", size=28, color="black"),
        xanchor="left",
        yanchor="middle"
    )

    return fig

def create_T_fix_fig(H_vals, T_fix_res, T, data=None, language='eng'):
    tr = _tr(language)
    H_kOe = np.asarray(H_vals, dtype=float) / 1000.0
    (f1, t1), (f2, t2) = T_fix_res
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)

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
        H_exp  = np.asarray(data[0], dtype=float)
        lf_exp = np.asarray(data[1], dtype=float)
        hf_exp = np.asarray(data[2], dtype=float)
        err_lf_exp = np.asarray(data[5], dtype=float)
        err_hf_exp = np.asarray(data[6], dtype=float)

        y_lf = lf_exp.copy()
        y_hf = hf_exp.copy()
        err_y_lf = err_lf_exp.copy()
        err_y_hf = err_hf_exp.copy()

        if T == 320:
            m_cross = H_exp >= 900
            tmp = y_lf[m_cross].copy()
            y_lf[m_cross] = y_hf[m_cross]
            y_hf[m_cross] = tmp
            err_tmp = err_y_lf[m_cross].copy()
            err_y_lf[m_cross] = err_y_hf[m_cross]
            err_y_hf[m_cross] = err_tmp

        fig.add_trace(go.Scatter(
            x=np.asarray(H_exp, dtype=float) / 1000.0, y=np.asarray(y_lf, dtype=float),
            mode='markers', name='LF' + tr['experiment_suffix'],
            error_y=dict(type="data", array=np.asarray(err_y_lf, dtype=float)),
            marker=dict(color=LF_COLOR, size=dot_size, line=dict(width=1, color="#000000"))
        ))
        fig.add_trace(go.Scatter(
            x=np.asarray(H_exp, dtype=float) / 1000.0, y=np.asarray(y_hf, dtype=float),
            mode='markers', name='HF' + tr['experiment_suffix'],
            error_y=dict(type="data", array=np.asarray(err_y_hf, dtype=float)),
            marker=dict(color=HF_COLOR, size=dot_size, line=dict(width=1, color="#000000"))
        ))

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman, Times, serif", size=14, color="black"),
        margin=dict(l=90, r=20, t=10, b=70),
        title=dict(
            text=tr['temperature_title'].format(value=f"{T:g}"),
            x=0.5, y=0.98,
            xref='paper',
            yref='paper',
            xanchor='center', yanchor='top',
            font=title_font
        ),
        xaxis=dict(
            title=dict(text=tr['magnetic_field_axis'], font=title_font, standoff=16),
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
            title=dict(text=tr['frequency_axis'], font=title_font, standoff=16),
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

def create_freq_fig(T_vals, H_vals, freq_res_grid, language='eng'):
    tr = _tr(language)
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
    x_plane = np.array([[Hmin, Hmax], [Hmin, Hmax]])
    y_plane = np.array([[T_plane, T_plane], [T_plane, T_plane]])
    z_plane = np.array([[zmin, zmin], [zmax, zmax]])

    title_font = dict(family="Times New Roman, Times, serif", size=18, color="black")
    tick_font  = dict(family="Times New Roman, Times, serif", size=14, color="black")

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=x_plane, y=y_plane, z=z_plane,
        colorscale=[[0, PLANE_COLOR], [1, PLANE_COLOR]],
        showscale=False,
        opacity=0.25,
        name=tr['temperature_title'].format(value=f"{T_plane:g}"),
        hoverinfo="skip"
    ))

    title_pad_lines = 10
    pad = "<br>" * int(title_pad_lines)

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
                title=dict(text=pad + tr['magnetic_field_axis'], font=title_font),
                tickfont=tick_font,
                tickcolor="black",
            ),
            yaxis=dict(
                title=dict(text=pad + tr['temperature_axis'], font=title_font),
                tickfont=tick_font,
                tickcolor="black",
            ),
            zaxis=dict(
                title=dict(text=pad + tr['frequency_axis'], font=title_font),
                tickfont=tick_font,
                tickcolor="black",
            ),
            camera=dict(projection=dict(type='orthographic')),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.7),
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

def create_phase_fig(T_vals, theta_0, language='eng'):
    tr = _tr(language)
    H_kOe = np.arange(0, 4001, 50) / 1000
    theta_0 = theta_0.T
    custom_colorscale = [
        [0.00, 'rgb(0, 0, 0)'],
        [0.31, 'rgb(0, 0, 255)'],
        [0.62, 'rgb(0, 128, 0)'],
        [0.93, 'rgb(255, 255, 0)'],
        [1.00, 'rgb(255, 255, 255)']
    ]
    heat = go.Heatmap(
        x=T_vals,
        y=H_kOe,
        z=theta_0,
        colorscale=custom_colorscale,
        colorbar=dict(
            title=dict(
                text=tr['phase_colorbar'],
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

    fig.update_layout(
        xaxis=dict(title=tr['phase_x'], tickcolor="black", range=[T_vals.min(), T_vals.max()]),
        yaxis=dict(title=tr['phase_y'], tickcolor="black", range=[H_kOe.min(), H_kOe.max()]),
        template='plotly_white',
        margin=dict(l=60, r=40, t=40, b=60)
    )

    return fig

__all__ = [
    'create_phi_fig', 'create_theta_fig', 'create_yz_fig', 'create_H_fix_fig', 'create_phase_fig',
    'create_T_fix_fig', 'create_freq_fig',
]
