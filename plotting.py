# plotting.py
import plotly.graph_objs as go
import numpy as np

def create_phi_fig(time, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=phi, mode='lines', name='Численное решение', line=dict(color='red')))
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
                f"<b>Магнитное поле</b> H = {H} Oe<br>"
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
        }]
    )
    return fig

def create_theta_fig(time, theta, theta_fit):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=theta, mode='lines', name='Численное решение', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=time, y=theta_fit, mode='lines', name='Аппроксимация', line=dict(color='blue', dash='dash')))
    fig.update_layout(
        title="Динамика угла θ",
        xaxis_title="Время (нс)",
        yaxis_title="Отклонение угла θ (градусы)",
        template="plotly_white"
    )
    return fig

def create_yz_fig(y, z, time, anim_speed=5):
    frames = []
    num_frames = len(time[::anim_speed])
    for i in range(num_frames):
        frame = go.Frame(
            data=[go.Scatter(
                x=y[: (i+1) * anim_speed],
                y=z[: (i+1) * anim_speed],
                mode='lines',
                line=dict(color='rgb(255, 0, 0)', width=2)
            )]
        )
        frames.append(frame)
    fig = go.Figure(
        data=[go.Scatter(x=y, y=z, mode='lines', line=dict(color='rgb(255, 0, 0)', width=2))],
        frames=frames
    )
    lim = np.max([np.abs(y), np.abs(z)])
    limits = (-1.1 * lim, 1.1 * lim)
    fig.update_layout(
        title="Проекция траектории на плоскость yz",
        xaxis_title="Координата y (норм.)",
        yaxis_title="Координата z (норм.)",
        xaxis=dict(range=limits),
        yaxis=dict(range=limits),
        template="plotly_white",
        updatemenus=[{
            "type": "buttons",
            "showactive": True,
            "x": 1,
            "y": 1,
            "xanchor": 'right',
            "yanchor": 'top',
            "direction": 'left',
            "buttons": [{
                "label": "Запуск",
                "method": "animate",
                "args": [None, {"frame": {"duration": 50, "redraw": True},
                                "fromcurrent": False, "mode": "immediate"}]
            }]
        }]
    )
    return fig

def create_H_fix_fig(T_vals, H_fix_freqs, H):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T_vals, y=H_fix_freqs[0], mode='lines', name='HF', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=T_vals, y=H_fix_freqs[1], mode='lines', name='LF', line=dict(color='red')))
    fig.update_layout(
        title=f"Зависимость частот от температуры при H = {H} Oe",
        xaxis_title="Температура (K)",
        yaxis_title="Частота (ГГц)",
        template="plotly_white"
    )
    return fig

def create_T_fix_fig(H_vals, T_fix_freqs, T):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=H_vals, y=T_fix_freqs[0], mode='lines', name='HF', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=H_vals, y=T_fix_freqs[1], mode='lines', name='LF', line=dict(color='red')))
    fig.update_layout(
        title=f"Зависимость частот от магнитного поля при T = {T} K",
        xaxis_title="Магнитное поле (Oe)",
        yaxis_title="Частота (ГГц)",
        template="plotly_white"
    )
    return fig
