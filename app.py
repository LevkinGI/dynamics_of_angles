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

from dataclasses import asdict
from config import SimParams
from constants import *
from simulator import run_simulation
# Функции аппроксимации из Cython-модуля:
from fitting_cy import fit_function_theta, fit_function_phi
from fitting import residuals_stage1, residuals_stage2, combined_residuals
from plotting import *

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    dcc.Store(
        id='param-store',
        data={
            "1": asdict(SimParams(1.0, 1.0, 1.0, 1.0, 1.0)),
            "2": asdict(SimParams(1.0, 1.0, 1.0, 1.0, 1.0)),
        }
    ),
    dcc.Store(id='freq-cache',  data=None),

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
        value=T_init,
        marks={i: str(i) for i in range(290, 351, 10)}
    ),
    html.Div(id='selected-T-value', style={'margin-bottom': '20px'}),

    html.Div([
        html.Div([
            html.Label("k × α"),
            dcc.Slider(id='alpha-scale-slider',
                       min=0.8, max=1.2, step=0.005, value=1.0,
                       marks={round(i/100, 2): f"{i/100:.2f}" for i in range(80, 121, 5)},
                       vertical=True, verticalHeight=180),
            ],
            style={"marginRight": "24px"}
        ),
        html.Div([
            html.Label("k × χ"),
            dcc.Slider(id='chi-scale-slider',
                       min=0.8, max=1.2, step=0.005, value=1.0,
                       marks={round(i/100, 2): f"{i/100:.2f}" for i in range(80, 121, 5)},
                       vertical=True, verticalHeight=180),
            ],
            style={"marginRight": "24px"}
        ),
        html.Div([
            html.Label("k × K(T)"),
            dcc.Slider(id='k-scale-slider',
                       min=0.8, max=1.2, step=0.005, value=1.0,
                       marks={round(i/100, 2): f"{i/100:.2f}" for i in range(80, 121, 5)},
                       vertical=True, verticalHeight=180),
            ],
            style={"marginRight": "24px"}
        ),
        html.Div([
            html.Label("k × m"),
            dcc.Slider(id='m-scale-slider',
                       min=0.8, max=1.2, step=0.005, value=1.0,
                       marks={round(i/100, 2): f"{i/100:.2f}" for i in range(80, 121, 5)},
                       vertical=True, verticalHeight=180),
            ],
            style={"marginRight": "24px"}
        ),
        html.Div([
            html.Label("k × M"),
            dcc.Slider(id='M-scale-slider',
                       min=0.8, max=1.2, step=0.005, value=1.0,
                       marks={round(i/100, 2): f"{i/100:.2f}" for i in range(80, 121, 5)},
                       vertical=True, verticalHeight=180),
            ]
        ),],
        style={
            "display":   "flex",
            "alignItems": "flex-start",   # вершины всех ползунков выровнены
            "flexWrap":  "nowrap"         # гарантирует одну строку
        },
    ),
    
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
                        xaxis_title='Магнитное поле (Э)',
                        yaxis_title='Температура (K)',
                        zaxis_title='Амплитуда φ (°)'
                    ),
                    font=dict(size=14),
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
                        xaxis_title='Магнитное поле (Э)',
                        yaxis_title='Температура (K)',
                        zaxis_title='Амплитуда θ (°)'
                    ),
                    font=dict(size=14),
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
                    go.Surface(z=f1_GHz, x=H_vals, y=T_vals_1,
                               colorscale=[[0, 'rgb(255, 182, 193)'], [1, 'rgb(255, 0, 0)']],
                               showscale=False, name='HF'),
                    go.Surface(z=f2_GHz, x=H_vals, y=T_vals_1,
                               colorscale=[[0, 'rgb(173, 216, 230)'], [1, 'rgb(0, 0, 255)']],
                               showscale=False, name='LF')
                ],
                layout=go.Layout(
                    title="Частоты LF и HF в зависимости от H и T",
                    scene=dict(
                        xaxis_title='Магнитное поле (Э)',
                        yaxis_title='Температура (K)',
                        zaxis_title='Частота (ГГц)'
                    ),
                    font=dict(size=14),
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
    return f'Магнитное поле H = {H} Э:', f'Температура T = {T} K:'

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
        T = T_init
    t_index = np.abs(t_vals - T).argmin()
    
    min_val = t_vals[0]
    max_val = t_vals[-1]
    step = np.round(t_vals[1] - t_vals[0], decimals=1) 
    value = t_vals[t_index]
    marks = {float(val): str(val) for val in t_vals if val % 10 == 0}
    return min_val, max_val, step, value, marks

@app.callback(
    [Output('alpha-scale-slider',      'value'),
    Output('chi-scale-slider',      'value'),
    Output('k-scale-slider',      'value'),
    Output('m-scale-slider',      'value'),
    Output('M-scale-slider',      'value')],
    Input('material-dropdown', 'value'),
    State('param-store',       'data'),    
)
def sync_sliders_with_material(material, store):
    p = SimParams(**store[material])
    return (p.alpha_scale, p.chi_scale,
            p.k_scale, p.m_scale, p.M_scale)

@app.callback(
    Output('param-store', 'data'),
    [Input('material-dropdown', 'value'),
    Input('alpha-scale-slider',      'value'),
    Input('chi-scale-slider',        'value'),
    Input('k-scale-slider',    'value'),
    Input('m-scale-slider',    'value'),
    Input('M-scale-slider',    'value')],
    State('param-store',       'data'),
)
def update_params(material, a_k, chi_k, k_k, m_k, M_k, store):
    p = SimParams(**store[material])
    p.alpha_scale = a_k
    p.chi_scale = chi_k
    p.k_scale = k_k
    p.m_scale = m_k
    p.M_scale = M_k
    store[material] = asdict(p)
    return store

@app.callback(
    Output('freq-cache', 'data'),
    [Input('param-store',       'data'),
     Input('material-dropdown', 'value')],
)
def update_freq_cache(store, material):
    p = SimParams(**store[material])

    h_index = np.abs(H_vals - 1000).argmin()
    T_vals = T_vals_1 if material=='1' else T_vals_2
    t_index = np.abs(T_vals - 293).argmin()

    H_mesh = H_mesh_1 if material == '1' else H_mesh_2
    T_mesh = T_mesh_1 if material == '1' else T_mesh_2
    m_mesh = p.m_scale * (m_mesh_1 if material == '1' else m_mesh_2)
    K_mesh = p.k_scale * (K_mesh_1 if material == '1' else K_mesh_2)
    chi_mesh = p.chi_scale * (chi_mesh_1 if material == '1' else chi_mesh_2)

    f1, f2 = compute_frequencies(H_mesh, m_mesh, chi_mesh, K_mesh, gamma)

    # 1) Что лежит в подготовленных сетках, которые ты
    #    передаёшь в compute_frequencies_numba из update_graphs?
    print("m_mesh:",   np.any(m_mesh - m_mesh_1 > 1e-8))
    print("K_mesh:",   np.any(K_mesh - K_mesh_1 > 1e-8))
    print("chi_mesh:", np.any(chi_mesh - chi_mesh_1 > 1e-12))
    print("H_mesh:",   np.any(H_mesh - H_mesh_1 != 0))
    print("T_mesh:",   np.any(T_mesh - T_mesh_1 != 0))
    print("m_mesh_1:",   m_mesh_1[t_index, h_index])
    print("K_mesh_1:",   K_mesh_1[t_index, h_index])
    print("chi_mesh_1:", chi_mesh_1[t_index, h_index])
    print("H_mesh_1:",   H_mesh_1[t_index, h_index])
    print("T_mesh_1:",   T_mesh_1[t_index, h_index])
    print("gamma:",    gamma)
    
    # 2) А теперь – ровно те же индексы (t_index, h_index)
    #    в уже посчитанном «хорошем» массиве, который создаётся в constants.py:
    print("f1_good:", f1_GHz[t_index, h_index])
    print("f2_good:", f2_GHz[t_index, h_index])
    
    # 3) А теперь – ровно те же индексы (t_index, h_index)
    print("f1_changed:", f1[t_index, h_index])
    print("f2_changed:", f2[t_index, h_index])
    
    return {
        "freq_array1": f1.tolist(),
        "freq_array2": f2.tolist()
    }

@app.callback(
    [Output('phi-graph', 'figure'),
     Output('theta-graph', 'figure'),
     Output('yz-graph', 'figure'),
     Output('H_fix-graph', 'figure'),
     Output('T_fix-graph', 'figure'),
     Output('phi-amplitude-graph', 'figure'),
     Output('theta-amplitude-graph', 'figure'),
     Output('frequency-surface-graph', 'figure')],
    [Input('param-store', 'data'),
     Input('freq-cache', 'data'),
     Input('H-slider', 'value'),
     Input('T-slider', 'value'),
     Input('material-dropdown', 'value')]
)
def update_graphs(store, freqs, H, T, material):
    p = SimParams(**store[material])

    if freqs is None:
        freq_array1, freq_array2 = f1_GHz, f2_GHz
    else:
        freq_array1 = np.array(freqs["freq_array1"])
        freq_array2 = np.array(freqs["freq_array2"])
    
    # Определяем, какой input вызвал callback
    ctx = callback_context
    triggered_inputs = [t['prop_id'] for t in ctx.triggered]
    material_changed = any('material-dropdown' in ti for ti in triggered_inputs)
    freqs_changed  = any('freq-cache' in ti for ti in triggered_inputs)
  
    h_index = np.abs(H_vals - H).argmin()
    
    # Выбор данных в зависимости от материала
    T_vals = T_vals_1 if material=='1' else T_vals_2
    t_index = np.abs(T_vals - T).argmin()
    m_val = p.m_scale * (m_array_1 if material=='1' else m_array_2)[t_index]
    M_val = p.M_scale * (M_array_1 if material=='1' else M_array_2)[t_index]
    chi_val = p.chi_scale * (chi_T(T) if material=='1' else chi_const)
    K_val = p.k_scale * (K_T(T) if material=='1' else K_const)
    alpha = p.alpha_scale * (alpha_1 if material=='1' else alpha_2)
    amplitude_phi_static = phi_amplitude if material=='1' else phi_amplitude_2
    amplitude_theta_static = theta_amplitude if material=='1' else theta_amplitude_2
    kappa = m_val / gamma
    
    theor_freqs_GHz = sorted(np.round([freq_array1[t_index, h_index], freq_array2[t_index, h_index]], 1), reverse=True)

    sim_time, sol = run_simulation(H, T, m_val, M_val, K_val, chi_val, alpha, kappa)

    time_ns = sim_time * 1e9
    theta = np.degrees(sol[0])
    phi = np.degrees(sol[1])

    # Выполнение аппроксимации
    if False:
        A1_theta = np.max(theta) / 2
        A2_theta = A1_theta
        A1_phi = np.max(phi) / 2
        A2_phi = A1_phi
    
        initial_guess_stage1 = [0, 2, 0, 2, 0, 2, 0, 2, theor_freqs_GHz[0], theor_freqs_GHz[1]]
        lower_bounds_stage1 = [-np.pi, 0.01, -np.pi, 0.01, -np.pi, 0.01, -np.pi, 0.01, 0.1, 0.1]
        upper_bounds_stage1 = [np.pi, 100, np.pi, 100, np.pi, 100, np.pi, 100, 120, 120]
    
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
            np.inf, np.pi, 100, np.inf, np.pi, 100, 120, 120
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

        phi_fig = create_phi_fig(time_ns, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz, material)
        theta_fig = create_theta_fig(time_ns, theta, theta_fit)

    else:
        approx_freqs_GHz = (None, None)
        theta_fit = None
        phi_fit = None
    
    # Далее строим графики
    phi_fig = create_phi_fig(time_ns, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz, material)
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
    elif freqs_changed:
        phi_amp_fig = no_update
        theta_amp_fig = no_update
        freq_fig = create_freq_fig(T_vals, H_vals, freq_array1, freq_array2)
    else:
        phi_amp_fig = no_update
        theta_amp_fig = no_update
        freq_fig = no_update

    return phi_fig, theta_fig, yz_fig, H_fix_fig, T_fix_fig, phi_amp_fig, theta_amp_fig, freq_fig

if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=8000)
