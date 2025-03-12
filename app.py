# app.py

# Автоматическая сборка Cython модулей при импорте
import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

import dash
from dash import dcc, html, no_update, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from scipy.optimize import least_squares

from constants import (
    H_vals, T_vals_1, T_vals_2, m_array, M_array, f1_GHz_1, f2_GHz_1, f1_GHz_2, f2_GHz_2,
    phi_amplitude, theta_amplitude, chi_T, K_T, gamma, alpha_1, alpha_2,
    m_array_2, M_array_2, phi_amplitude_2, theta_amplitude_2, K_const, chi_const
)
from simulator import run_simulation
# Функции аппроксимации из Cython-модуля:
from fitting_cy import fit_function_theta, fit_function_phi
from fitting import residuals_stage1, residuals_stage2, combined_residuals
from plotting import (
    create_phi_fig, create_theta_fig, create_yz_fig, create_H_fix_fig,
    create_T_fix_fig, create_phi_amp_fig, create_theta_amp_fig, create_freq_fig
)

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
    dcc.Dropdown(
        id='material-dropdown',
        options=[
            {'label': 'FeFe', 'value': '1'},
            {'label': 'GdFe', 'value': '2'}
        ],
        value='1',
        style={'width': '300px'}
    ),
    html.Div([
        dcc.Graph(id='phi-graph', style={'display': 'inline-block', 'width': '50%'}),
        dcc.Graph(id='theta-graph', style={'display': 'inline-block', 'width': '50%'})
    ]),
    html.Div([
        dcc.Graph(
            id='phi-amplitude-graph',
            style={'display': 'inline-block', 'width': '33%', 'height': 'calc(33vw)'},
            figure=go.Figure(
                data=[go.Surface(z=phi_amplitude, x=H_vals, y=T_vals_1,
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
                data=[go.Surface(z=theta_amplitude, x=H_vals, y=T_vals_1,
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
                    go.Surface(z=f1_GHz_1, x=H_vals, y=T_vals_1,
                               colorscale=[[0, 'rgb(173, 216, 230)'], [1, 'rgb(0, 0, 255)']],
                               showscale=False, name='HF'),
                    go.Surface(z=f2_GHz_1, x=H_vals, y=T_vals_1,
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
    [Output('T-slider', 'min'),
     Output('T-slider', 'max'),
     Output('T-slider', 'step'),
     Output('T-slider', 'value'),
     Output('T-slider', 'marks')],
    [Input('material-dropdown', 'value')],
    [State('T-slider', 'value')]
)
def update_T_slider(material, T):
    if material == '1':
        t_vals = T_vals_1
    else:
        t_vals = T_vals_2
    if T is None:
        T = 300
    t_index = np.abs(t_vals - T).argmin()
    
    min_val = t_vals[0]
    max_val = t_vals[-1]
    step = np.round(t_vals[1] - t_vals[0], decimals=1) 
    value = t_vals[t_index]
    marks = {val: str(val) for val in np.round(t_vals, decimals=1) if val % 10 == 0}
    return min_val, max_val, step, value, marks

@app.callback(
    [Output('phi-graph', 'figure'),
     Output('theta-graph', 'figure'),
     Output('yz-graph', 'figure'),
     Output('H_fix-graph', 'figure'),
     Output('T_fix-graph', 'figure'),
     Output('phi-amplitude-graph', 'figure'),
     Output('theta-amplitude-graph', 'figure'),
     Output('frequency-surface-graph', 'figure')],
    [Input('H-slider', 'value'),
     Input('T-slider', 'value'),
     Input('material-dropdown', 'value')]
)
def update_graphs(H, T, material):
    # Определяем, какой input вызвал callback
    ctx = callback_context
    triggered_inputs = [t['prop_id'] for t in ctx.triggered]
    material_changed = any('material-dropdown' in ti for ti in triggered_inputs)
  
    h_index = np.abs(H_vals - H).argmin()
    
    # Выбор данных в зависимости от материала
    if material == '1':
        T_vals = T_vals_1
        t_index = np.abs(T_vals - T).argmin()
        m_val = m_array[t_index]
        M_val = M_array[t_index]
        chi_val = chi_T(T)
        K_val = K_T(T)
        alpha = alpha_1
        amplitude_phi_static = phi_amplitude
        amplitude_theta_static = theta_amplitude
        freq_array1 = f1_GHz_1
        freq_array2 = f2_GHz_1
    else:  # материал 2
        T_vals = T_vals_2
        t_index = np.abs(T_vals - T).argmin()
        m_val = m_array_2[t_index]
        M_val = M_array_2[t_index]
        chi_val = chi_const
        K_val = K_const
        alpha = alpha_2
        amplitude_phi_static = phi_amplitude_2
        amplitude_theta_static = theta_amplitude_2
        freq_array1 = f1_GHz_2
        freq_array2 = f2_GHz_2
    kappa = m_val / gamma
    theor_freqs_GHz = sorted(np.round([freq_array1[t_index, h_index], freq_array2[t_index, h_index]], 1), reverse=True)

    sim_time, sol = run_simulation(H, T, m_val, M_val, chi_val, K_val, kappa, alpha)

    time_ns = sim_time * 1e9
    theta = np.degrees(sol[0])
    phi = np.degrees(sol[1])

    # Выполнение аппроксимации
    A1_theta = np.max(theta) / 2
    A2_theta = A1_theta
    A1_phi = np.max(phi) / 2
    A2_phi = A1_phi

    initial_guess_stage1 = [0, 2, 0, 2, 0, 2, 0, 2, theor_freqs_GHz[0], theor_freqs_GHz[1]]
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

    initial_guess_stage2 = [A1_theta, A2_theta, A1_phi, A2_phi]
    result_stage2 = least_squares(
        residuals_stage2,
        x0=initial_guess_stage2,
        args=(sim_time, theta, phi, f1_theta_opt, n1_theta_opt, f2_theta_opt, n2_theta_opt,
              f1_phi_opt, n1_phi_opt, f2_phi_opt, n2_phi_opt, f1_GHz_opt, f2_GHz_opt),
        xtol=1e-4, ftol=1e-4, gtol=1e-4, max_nfev=10000
    )
    A1_theta_opt, A2_theta_opt, A1_phi_opt, A2_phi_opt = result_stage2.x

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

    theta_fit = fit_function_theta(sim_time, A1_theta_opt, f1_theta_opt, n1_theta_opt,
                               A2_theta_opt, f2_theta_opt, n2_theta_opt,
                               f1_GHz_opt, f2_GHz_opt)
    phi_fit = fit_function_phi(sim_time, A1_phi_opt, f1_phi_opt, n1_phi_opt,
                               A2_phi_opt, f2_phi_opt, n2_phi_opt,
                               f1_GHz_opt, f2_GHz_opt)
    
    approx_freqs_GHz = sorted(np.round([f1_GHz_opt, f2_GHz_opt], 1), reverse=True)
    
    # Далее строим графики
    phi_fig = create_phi_fig(time_ns, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz)
    theta_fig = create_theta_fig(time_ns, theta, theta_fit)
    yz_fig = create_yz_fig(np.sin(np.pi/2 + np.radians(theta)) * np.sin(np.radians(phi)),
                           np.cos(np.pi/2 + np.radians(theta)),
                           time_ns)
    H_fix_freqs = (freq_array1[:, h_index], freq_array2[:, h_index])
    T_fix_freqs = (freq_array1[t_index, :], freq_array2[t_index, :])
    H_fix_fig = create_H_fix_fig(T_vals, H_fix_freqs, H)
    T_fix_fig = create_T_fix_fig(H_vals, T_fix_freqs, T)
    if material_changed:
        phi_amp_fig = create_phi_amp_fig(T_vals, H_vals, amplitude_phi_static)
        theta_amp_fig = create_theta_amp_fig(T_vals, H_vals, amplitude_theta_static)
        freq_fig = create_freq_fig(T_vals, H_vals, freq_array1, freq_array2)
    else:
        phi_amp_fig = no_update
        theta_amp_fig = no_update
        freq_fig = no_update

    return phi_fig, theta_fig, yz_fig, H_fix_fig, T_fix_fig, phi_amp_fig, theta_amp_fig, freq_fig

if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=8000)
