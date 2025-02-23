# app.py

# Автоматическая сборка Cython модулей при импорте
import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from scipy.optimize import least_squares

from constants import (H_vals, T_vals, m_array, M_array, f1_GHz, f2_GHz, phi_amplitude, theta_amplitude, chi_T, K_T, gamma, 
                       m_array_2, M_array_2, phi_amplitude_2, theta_amplitude_2, K_const, chi_const)
from simulator import run_simulation
# Импорт функций аппроксимации из Cython модуля вместо Python-версии:
from fitting_cy import fit_function_theta, fit_function_phi
from fitting import residuals_stage1, residuals_stage2, combined_residuals
from plotting import create_phi_fig, create_theta_fig, create_yz_fig, create_H_fix_fig, create_T_fix_fig, create_phi_amp_fig, create_theta_amp_fig

# Простой кэш для результатов симуляции по выбранным значениям H и T
simulation_cache = {}

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Динамика углов θ и φ при различных значениях магнитного поля и температуры"),
    html.Label(id='H-label'),
    dcc.Slider(
        id='H-slider',
        min=0,
        max=H_vals[-1],
        step=10,
        value=1000,
        marks={i: str(i) for i in range(0, H_vals[-1] + 1, 500)}
    ),
    html.Div(id='selected-H-value', style={'margin-bottom': '20px'}),
    html.Label(id='T-label'),
    dcc.Slider(
        id='T-slider',
        min=290,
        max=350,
        step=0.1,
        value=300,
        marks={i: str(i) for i in range(290, 351, 10)}
    ),
    html.Div(id='selected-T-value', style={'margin-bottom': '20px'}),
    html.Div([
        dcc.Graph(id='phi-graph', style={'display': 'inline-block', 'width': '50%'}),
        dcc.Graph(id='theta-graph', style={'display': 'inline-block', 'width': '50%'})
    ]),
    html.Div([
        dcc.Graph(
            id='phi-amplitude-graph',
            style={'display': 'inline-block', 'width': '33%', 'height': 'calc(33vw)'},
            figure=go.Figure(
                data=[go.Surface(z=phi_amplitude, x=H_vals, y=T_vals,
                                 colorscale=[[0, 'rgb(173, 216, 230)'], [1, 'rgb(0, 0, 255)']],
                                 showscale=False, name='LF')],
                layout=go.Layout(
                    scene=dict(
                        xaxis_title='Магнитное поле (Oe)',
                        yaxis_title='Температура (K)',
                        zaxis_title='Амплитуда φ (°)'
                    ),
                    template="plotly_white"
                )
            )
        ),
        dcc.Graph(
            id='theta-amplitude-graph',
            style={'display': 'inline-block', 'width': '33%', 'height': 'calc(33vw)'},
            figure=go.Figure(
                data=[go.Surface(z=theta_amplitude, x=H_vals, y=T_vals,
                                 colorscale=[[0, 'rgb(255, 182, 193)'], [1, 'rgb(255, 0, 0)']],
                                 showscale=False, name='HF')],
                layout=go.Layout(
                    scene=dict(
                        xaxis_title='Магнитное поле (Oe)',
                        yaxis_title='Температура (K)',
                        zaxis_title='Амплитуда θ (°)'
                    ),
                    template="plotly_white"
                )
            )
        ),
        dcc.Graph(id='yz-graph', style={'display': 'inline-block', 'width': '33%', 'height': 'calc(33vw)'})
    ]),
    html.Div([
        dcc.Graph(
            id='frequency-surface-graph',
            style={'display': 'inline-block', 'width': '33%', 'height': 'calc(33vw)'},
            figure=go.Figure(
                data=[
                    go.Surface(z=f1_GHz, x=H_vals, y=T_vals,
                               colorscale=[[0, 'rgb(173, 216, 230)'], [1, 'rgb(0, 0, 255)']],
                               showscale=False, name='HF'),
                    go.Surface(z=f2_GHz, x=H_vals, y=T_vals,
                               colorscale=[[0, 'rgb(255, 182, 193)'], [1, 'rgb(255, 0, 0)']],
                               showscale=False, name='LF')
                ],
                layout=go.Layout(
                    title="Частоты LF и HF в зависимости от H и T",
                    scene=dict(
                        xaxis_title='Магнитное поле (Oe)',
                        yaxis_title='Температура (K)',
                        zaxis_title='Частота (ГГц)'
                    ),
                    template="plotly_white"
                )
            )
        ),
        dcc.Graph(id='H_fix-graph', style={'display': 'inline-block', 'width': '33%', 'height': 'calc(33vw)'}),
        dcc.Graph(id='T_fix-graph', style={'display': 'inline-block', 'width': '33%', 'height': 'calc(33vw)'})
    ])
])

@app.callback(
    [Output('H-label', 'children'),
     Output('T-label', 'children')],
    [Input('H-slider', 'value'),
     Input('T-slider', 'value')]
)
def update_slider_values(H, T):
    return f'Магнитное поле H = {H} Oe:', f'Температура T = {T} K:'

@app.callback(
    [Output('phi-graph', 'figure'),
     Output('theta-graph', 'figure'),
     Output('yz-graph', 'figure'),
     Output('H_fix-graph', 'figure'),
     Output('T_fix-graph', 'figure')],
    [Input('H-slider', 'value'),
     Input('T-slider', 'value')]
)
def update_graphs(H, T, material):
    # Определяем, какой input вызвал callback
    ctx = callback_context
    triggered_inputs = [t['prop_id'] for t in ctx.triggered]
    matereal_changed = any('material-dropdown' in ti for ti in triggered_inputs)
  
    t_index = np.abs(T_vals - T).argmin()
    h_index = np.abs(H_vals - H).argmin()
    
    # Выбор данных в зависимости от материала
    if material == '1':
        m_val = m_array[t_index]
        M_val = M_array[t_index]
        chi_val = chi_T(T)
        K_val = K_T(T)
        amplitude_phi_static = phi_amplitude
        amplitude_theta_static = theta_amplitude
        # Для материала 1 можно использовать предвычисленные частотные массивы f1_GHz, f2_GHz
        freq_array1 = f1_GHz
        freq_array2 = f2_GHz
    else:  # материал 2
        m_val = m_array_2[t_index]
        M_val = M_array_2[t_index]
        chi_val = chi_const    # константное значение
        K_val = K_const
        amplitude_phi_static = phi_amplitude_2
        amplitude_theta_static = theta_amplitude_2
    
    kappa = m_val / gamma

    # Симуляция с кэшированием
    sim_key = (H, T, material)
    if sim_key in simulation_cache:
        sim_time, sol = simulation_cache[sim_key]
    else:
        sim_time, sol = run_simulation(H, T, m_val, M_val, chi_val, K_val, kappa)
        simulation_cache[sim_key] = (sim_time, sol)
    time_ns = sim_time * 1e9
    theta = np.degrees(sol[0])
    phi = np.degrees(sol[1])

    # Этап 1: Оптимизация параметров (без амплитуд)
    A1_theta = np.max(theta) / 2
    A2_theta = np.max(theta) / 2
    A1_phi = np.max(phi) / 2
    A2_phi = np.max(phi) / 2

    initial_guess_stage1 = [0, 2, 0, 2, 0, 2, 0, 2, 50, 10]
    lower_bounds_stage1 = [-np.pi, 0.01, -np.pi, 0.01, -np.pi, 0.01, -np.pi, 0.01, 0.1, 0.1]
    upper_bounds_stage1 = [np.pi, 100, np.pi, 100, np.pi, 100, np.pi, 100, 100, 100]

    result_stage1 = least_squares(
        residuals_stage1,
        x0=initial_guess_stage1,
        bounds=(lower_bounds_stage1, upper_bounds_stage1),
        args=(sim_time, theta, phi, A1_theta, A2_theta, A1_phi, A2_phi),
        xtol=1e-4, ftol=1e-4, gtol=1e-4, max_nfev=10000
    )
    (f1_theta_opt, n1_theta_opt, f2_theta_opt, n2_theta_opt,
     f1_phi_opt, n1_phi_opt, f2_phi_opt, n2_phi_opt,
     f1_GHz_opt, f2_GHz_opt) = result_stage1.x

    # Этап 2: Оптимизация амплитуд и фаз
    initial_guess_stage2 = [A1_theta, A2_theta, A1_phi, A2_phi]
    result_stage2 = least_squares(
        residuals_stage2,
        x0=initial_guess_stage2,
        args=(sim_time, theta, phi, f1_theta_opt, n1_theta_opt, f2_theta_opt, n2_theta_opt,
              f1_phi_opt, n1_phi_opt, f2_phi_opt, n2_phi_opt, f1_GHz_opt, f2_GHz_opt),
        xtol=1e-4, ftol=1e-4, gtol=1e-4, max_nfev=10000
    )
    A1_theta_opt, A2_theta_opt, A1_phi_opt, A2_phi_opt = result_stage2.x

    # Этап 3: Финальная оптимизация всех параметров
    initial_guess_stage3 = [
        A1_theta_opt, f1_theta_opt, n1_theta_opt, A2_theta_opt, f2_theta_opt, n2_theta_opt,
        A1_phi_opt, f1_phi_opt, n1_phi_opt, A2_phi_opt, f2_phi_opt, n2_phi_opt,
        f1_GHz_opt, f2_GHz_opt
    ]
    lower_bounds_stage3 = [
        -np.inf, -np.pi, 0.01, -np.inf, -np.pi, 0.01,
        -np.inf, -np.pi, 0.01, -np.inf, -np.pi, 0.01, 0.1, 0.1
    ]
    upper_bounds_stage3 = [
        np.inf, np.pi, 100, np.inf, np.pi, 100,
        np.inf, np.pi, 100, np.inf, np.pi, 100, 100, 100
    ]
    result_stage3 = least_squares(
        combined_residuals,
        x0=initial_guess_stage3,
        bounds=(lower_bounds_stage3, upper_bounds_stage3),
        args=(sim_time, theta, phi),
        xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=10000
    )
    opt_params = result_stage3.x
    (A1_theta_opt, f1_theta_opt, n1_theta_opt, A2_theta_opt, f2_theta_opt, n2_theta_opt,
     A1_phi_opt, f1_phi_opt, n1_phi_opt, A2_phi_opt, f2_phi_opt, n2_phi_opt,
     f1_GHz_opt, f2_GHz_opt) = opt_params

    # Вычисление аппроксимирующих функций
    theta_fit = fit_function_theta(sim_time, A1_theta_opt, f1_theta_opt, n1_theta_opt,
                                   A2_theta_opt, f2_theta_opt, n2_theta_opt,
                                   f1_GHz_opt, f2_GHz_opt)
    phi_fit = fit_function_phi(sim_time, A1_phi_opt, f1_phi_opt, n1_phi_opt,
                               A2_phi_opt, f2_phi_opt, n2_phi_opt,
                               f1_GHz_opt, f2_GHz_opt)

    approx_freqs = sorted([f1_GHz_opt, f2_GHz_opt], reverse=True)
    approx_freqs_GHz = [np.round(f, 1) for f in approx_freqs]
    theor_freqs_GHz = sorted(np.round([f1_GHz[t_index, h_index], f2_GHz[t_index, h_index]], 1), reverse=True)

    # Вычисление проекции траектории на плоскость yz
    theta_rad = np.pi / 2 + np.radians(theta)
    phi_rad = np.radians(phi)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)

    # Построение графиков зависимостей
    H_fix_freqs = (f1_GHz[t_index, :], f2_GHz[t_index, :])
    T_fix_freqs = (f1_GHz[:, h_index], f2_GHz[:, h_index])

    phi_fig = create_phi_fig(time, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz)
    theta_fig = create_theta_fig(time, theta, theta_fit)
    yz_fig = create_yz_fig(y, z, time)
    H_fix_fig = create_H_fix_fig(T_vals, H_fix_freqs, H)
    T_fix_fig = create_T_fix_fig(H_vals, T_fix_freqs, T)
    if matereal_changed:
        phi_amp_fig = create_phi_amp_fig(amplitude_phi_static)
        theta_amp_fig = create_theta_amp_fig(amplitude_theta_static)
    else:
        phi_amp_fig = no_update
        theta_amp_fig = no_update
        freq_fig = no_update
    return phi_fig, theta_fig, yz_fig, H_fix_fig, T_fix_fig, phi_amp_fig, theta_amp_fig, freq_fig = no_update

if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=8000)
