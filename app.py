# app.py

# Автоматическая сборка Cython модулей при импорте
import pyximport
import numpy as np
import time
import io
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

import dash
from dash import dcc, html, no_update, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_daq as daq
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
from copy import deepcopy

w_fix, h_fix = 600, 400
w_freq, h_freq = 600, 600
w_angles, h_angles = 900, 400
w_phase, h_phase = 600, 600
w_yz, h_yz = 520, 400

sliders_range = 5
log_marks = {}
for i in range(1, sliders_range + 1):
    if i > 10 and i % 10 != 0:
        continue
    v = float(np.log10(i))
    log_marks[f"{v:g}"]  = str(i)
    log_marks[f"{-v:g}"] = f"1/{i}"

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    dcc.Store(
        id='param-store',
        data={
            "1": asdict(SimParams(0.7, 1.0, 1.0, 1.0, 0.4)),
            "2": asdict(SimParams(1.0, 1.0, 1.0, 1.0, 1.0)),
        }
    ),

    html.H1("Динамика углов θ и φ при различных значениях магнитного поля и температуры"),
    html.Label(id='H-label'),
    dcc.Slider(
        id='H-slider',
        min=0,
        max=H_vals[-1],
        step=10,
        value=1700,
        marks={str(i): str(i) for i in range(0, int(H_vals[-1]) + 1, 500)},
        tooltip={"placement": "bottom", "always_visible": False}, updatemode="mouseup",
    ),
    html.Div(id='selected-H-value', style={'margin-bottom': '20px'}),
    html.Label(id='T-label'),
    dcc.Slider(
        id='T-slider',
        min=290,
        max=350,
        step=0.1,
        value=T_init,
        marks={str(i): str(i) for i in range(290, 351, 10)},
        tooltip={"placement": "bottom", "always_visible": False}, updatemode="mouseup",
    ),
    html.Div(id='selected-T-value', style={'margin-bottom': '20px'}),




    
    html.Div([
        html.Div([
            html.Label(id='alpha-scale-label'),
            dcc.Slider(id='alpha-scale-slider',
                       min=-np.log10(sliders_range), max=np.log10(sliders_range), step=0.001, value=np.log10(0.7),
                       marks=log_marks,
                       updatemode="mouseup",
                       vertical=True, verticalHeight=180,
                      ),
            ],
            style={"marginLeft": "30px","position": "relative"}
            ),
        
        html.Div([
            html.Label(id='k-scale-label'),
            dcc.Slider(id='k-scale-slider',
                       min=-np.log10(sliders_range), max=np.log10(sliders_range), step=0.001, value=0.0,
                       marks=log_marks,
                       updatemode="mouseup",
                       vertical=True, verticalHeight=180,
                      ),
            ],
            style={"marginLeft": "30px", "position": "relative"}
        ),
        
        html.Div([
            html.Label(id='m-scale-label'),
            dcc.Slider(id='m-scale-slider',
                       min=-np.log10(sliders_range), max=np.log10(sliders_range), step=0.001, value=0.0,
                       marks=log_marks,
                       updatemode="mouseup",
                       vertical=True, verticalHeight=180,
                      ),
            ],
            style={"marginLeft": "30px", "position": "relative"}
        ),
        
        html.Div([
            html.Label(id='M-scale-label'),
            dcc.Slider(id='M-scale-slider',
                       min=-np.log10(sliders_range), max=np.log10(sliders_range), step=0.001, value=0.0,
                       marks=log_marks,
                       updatemode="mouseup",
                       vertical=True, verticalHeight=180,
                      ),
            ],
            style={"marginLeft": "30px", "position": "relative"}
        ),
        
        html.Div([
            html.Label(id='lam-scale-label'),
            dcc.Slider(id='lam-scale-slider',
                       min=-np.log10(sliders_range), max=np.log10(sliders_range), step=0.001, value=np.log10(0.4),
                       marks=log_marks,
                       updatemode="mouseup",
                       vertical=True, verticalHeight=180,
                      ),
            ],
            style={"marginLeft": "30px", "position": "relative"}
        ),

        html.Div([
            daq.BooleanSwitch(
                id='auto-calc-switch',
                on=False,
                label='Моделирование при изменении параметров',
                labelPosition='top',
                color='#119DFF',
                style={"marginLeft": "60px"}
                ),
            daq.BooleanSwitch(
                id='exp-view-switch',
                on=True,
                label='Отображать экспериментальные данные',
                labelPosition='top',
                color='#119DFF',
                style={"marginLeft": "60px"}
                ),
            daq.BooleanSwitch(
                id='png-svg-switch',
                on=False,
                label='Скачивать картинки в векторном формате .svg',
                labelPosition='top',
                color='#119DFF',
                style={"marginLeft": "60px"}
                ),
            ],
            style={"marginLeft": "30px", "position": "relative"}
        ),
        ],
        style={
            "display":   "flex",
            "alignItems": "flex-start",   # вершины всех ползунков выровнены
            "flexWrap":  "nowrap",         # гарантирует одну строку
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
        dcc.Graph(
            id='phase-graph',
            style={'display': 'inline-block', 'verticalAlign': 'top',
                   'width': '25%', 'height': 'calc(19vw)'},
            config={
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'ФД',
                    'width': w_phase, 'height': h_phase,
                    'scale': 2
                }
            },
        ),
        dcc.Graph(
            id='frequency-surface-graph',
            style={'display': 'inline-block', 'verticalAlign': 'top',
                   'width': '25%', 'height': 'calc(25vw)'},
            config={
                'toImageButtonOptions': {
                    'format': 'png',       # Формат: 'svg', 'png', 'jpeg', 'webp'
                    'filename': 'Частоты 3D', # Имя файла при скачивании
                    'width': w_freq, 'height': h_freq,
                    'scale': 5             # Масштаб (для растра), для svg не так важно
                }
            },
        ),
        html.Div([
            html.Button('Скачать кривые', id='download-H-btn',
                        style={'margin-bottom': '5px'}),
            dcc.Download(id='download-H-file'),
            dcc.Graph(
                id='H_fix-graph',
                config={
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'H_fix',
                        'width': w_fix, 'height': h_fix,
                        'scale': 2
                    }
                },
            ),
        ], style={'display': 'inline-block', 'verticalAlign': 'top',
                  'width': '25%', 'height': 'calc(25vw)'}),
        html.Div([
            html.Button('Скачать кривые', id='download-T-btn',
                        style={'margin-bottom': '5px'}),
            dcc.Download(id='download-T-file'),
            dcc.Graph(
                id='T_fix-graph',
                config={
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'T_fix',
                        'width': w_fix, 'height': h_fix,
                        'scale': 2
                    }
                },
            ),
        ], style={'display': 'inline-block', 'verticalAlign': 'top',
                  'width': '25%', 'height': 'calc(25vw)'}),
    ]),



    
    
    html.Div([
        html.Div([
            dcc.Graph(id='phi-graph', config={'toImageButtonOptions': {'format': 'png','filename': 'Динамика phi','width': w_angles,'height': h_angles,'scale': 2}}),
            dcc.Graph(id='theta-graph', config={'toImageButtonOptions': {'format': 'png','filename': 'Динамика theta','width': w_angles,'height': h_angles,'scale': 2}}),
        ], style={'display': 'inline-block', 'verticalAlign': 'top',
                  'width': '60%', 'height': 'calc(40vw)'}),
        dcc.Graph(id='yz-graph', style={'display': 'inline-block', 'width': '40%', 'height': 'calc(40vw)'}, config={'toImageButtonOptions': {'format': 'png','filename': 'Проекция траектории','width': w_yz,'height': h_yz,'scale': 3}})
    ]),
])

@app.callback(
    [Output('H-label', 'children'),
     Output('T-label', 'children')],
    [Input('H-slider', 'value'),
     Input('T-slider', 'value')],
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
    [State('T-slider', 'value')],
    prevent_initial_call=True,
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
    marks = {f"{float(val):g}": str(val) for val in t_vals if val % 10 == 0}
    return min_val, max_val, step, value, marks
    
@app.callback(
    Output("alpha-scale-label", "children"),
    Input("alpha-scale-slider", "drag_value"),
    prevent_initial_call=True,
)
def move_alpha_slider(logk):
    if logk is None:                          # при первом рендере drag_value == None
        raise PreventUpdate
    k = 10**logk
    return f"{k:.2f} × α"

@app.callback(
    Output("k-scale-label", "children"),
    Input("k-scale-slider", "drag_value"),
    prevent_initial_call=True,
)
def move_k_slider(logk):
    if logk is None:
        raise PreventUpdate
    k = 10**logk
    return f"{k:.2f} × K(T)"

@app.callback(
    Output("m-scale-label", "children"),
    Input("m-scale-slider", "drag_value"),
    prevent_initial_call=True,
)
def move_m_slider(logk):
    if logk is None:
        raise PreventUpdate
    k = 10**logk
    return f"{k:.2f} × m"

@app.callback(
    Output("M-scale-label", "children"),
    Input("M-scale-slider", "drag_value"),
    prevent_initial_call=True,
)
def move_M_slider(logk):
    if logk is None:
        raise PreventUpdate
    k = 10**logk
    return f"{k:.2f} × M"

@app.callback(
    Output("lam-scale-label", "children"),
    Input("lam-scale-slider", "drag_value"),
    prevent_initial_call=True,
)
def move_lam_slider(logk):
    if logk is None:
        raise PreventUpdate
    k = 10**logk
    return f"{k:.2f} × λ"

@app.callback(
    [Output('H_fix-graph', 'figure'),
     Output('T_fix-graph', 'figure'),
     Output('phase-graph', 'figure'),
     Output('frequency-surface-graph', 'figure')],
    [Input('H-slider', 'value'),
    Input('T-slider', 'value'),
    Input("alpha-scale-slider", "value"),
    Input("k-scale-slider", "value"),
    Input("m-scale-slider", "value"),
    Input("M-scale-slider", "value"),
    Input("lam-scale-slider", "value"),
    Input('material-dropdown', 'value'),
    Input('exp-view-switch',  'on'),
    Input('png-svg-switch', 'on'),],
    [State('H_fix-graph', 'figure'),
     State('T_fix-graph', 'figure'),
     State('phase-graph', 'figure'),
     State('frequency-surface-graph', 'figure'),],
)
def live_fix_graphs(H, T, a_val, k_val, m_val, M_val, lam_val, material, exp_on, svg_on, H_fix_fig, T_fix_fig, phase_fig, freq_fig):
    ctx = callback_context
    triggered_inputs = [t['prop_id'] for t in ctx.triggered]
    switch_on = any('png-svg-switch' in ti for ti in triggered_inputs)
    if switch_on: return H_fix_fig, T_fix_fig, phase_fig, freq_fig
    
    alpha_scale = 10**a_val
    k_scale     = 10**k_val
    m_scale     = 10**m_val
    M_scale     = 10**M_val
    lam_scale   = 10**lam_val
    
    T_vals  = T_vals_1 if material == '1' else T_vals_2
    t_index = np.abs(T_vals - T).argmin()
    m_array = m_scale * (m_array_1 if material == '1' else m_array_2)
    M_array = M_scale * (M_array_1 if material == '1' else M_array_2)
    K_array = k_scale * (K_array_1 if material == '1' else K_array_2)
    alpha   = alpha_scale * (alpha_1 if material == '1' else alpha_2)
    lam     = lam_scale * (lam_1 if material == '1' else lam_2)

    m_T = m_array[t_index]
    M_T = M_array[t_index]
    K_T = K_array[t_index]

    H_fix_res = compute_frequencies_H_fix(H, m_array, M_array, K_array, gamma, alpha, lam)
    T_fix_res = compute_frequencies_T_fix(H_vals, m_T, M_T, K_T, gamma, alpha, lam)
    freq_res_grid = compute_frequencies(H_vals[::2], m_array[4::6], M_array[4::6], K_array[4::6], gamma, alpha, lam)
    theta_0 = compute_phases(m_array[::6], M_array[::6], K_array[::6], lam)
    
    if material == '1' and exp_on:
        if T==293: T_data = T_293
        elif T==298: T_data = T_298
        elif T==308: T_data = T_308
        elif T==310: T_data = T_310
        elif T==320: T_data = T_320
        elif T==323: T_data = T_323
        else: T_data = None
        if H==1000: H_data = H_1000 
        elif H==1700: H_data = H_1700
        else: H_data = None
    else:
        T_data = None
        H_data = None
    
    H_fix_fig = create_H_fix_fig(T_vals, H_fix_res, H, data=H_data)
    T_fix_fig = create_T_fix_fig(H_vals, T_fix_res, T, data=T_data)
    freq_fig = create_freq_fig(T_vals[4::6], H_vals[::2], freq_res_grid)
    phase_fig = create_phase_fig(T_vals[::6], theta_0)

    return H_fix_fig, T_fix_fig, phase_fig, freq_fig

@app.callback(
    [Output('alpha-scale-slider',      'value'),
    Output('k-scale-slider',      'value'),
    Output('m-scale-slider',      'value'),
    Output('M-scale-slider',      'value'),
    Output('lam-scale-slider',      'value')],
    Input('material-dropdown', 'value'),
    State('param-store',       'data'),
)
def sync_sliders_with_material(material, store):
    p = SimParams(**store[material])
    return np.log10(p.alpha_scale), np.log10(p.k_scale), np.log10(p.m_scale), np.log10(p.M_scale), np.log10(p.lam_scale)

@app.callback(
    Output('param-store', 'data'),
    [Input('material-dropdown', 'value'),
    Input('alpha-scale-slider',      'value'),
    Input('k-scale-slider',    'value'),
    Input('m-scale-slider',    'value'),
    Input('M-scale-slider',    'value'),
    Input('lam-scale-slider',    'value')],
    State('param-store',       'data'),
    prevent_initial_call=True,
)
def update_params(material, a_k, k_k, m_k, M_k, lam_k, store):
    p = SimParams(**store[material])
    p.alpha_scale = 10 ** a_k
    p.k_scale = 10 ** k_k
    p.m_scale = 10 ** m_k
    p.M_scale = 10 ** M_k
    p.lam_scale = 10 ** lam_k
    store[material] = asdict(p)
    return store

@app.callback(
    [Output('phi-graph', 'figure'),
     Output('theta-graph', 'figure'),
     Output('yz-graph', 'figure'),],
    [Input('param-store', 'data'),
     Input('H-slider', 'value'),
     Input('T-slider', 'value'),
     Input('material-dropdown', 'value'),
     Input('auto-calc-switch',  'on'),
     Input('png-svg-switch', 'on'),],
    [State('phi-graph', 'figure'),
     State('theta-graph', 'figure'),
     State('yz-graph', 'figure')],
    prevent_initial_call=True,
)
def update_graphs(store, H, T, material, calc_on, svg_on, phi_fig, theta_fig, yz_fig):
    if not calc_on: raise PreventUpdate
        
    # Определяем, какой input вызвал callback
    ctx = callback_context
    triggered_inputs = [t['prop_id'] for t in ctx.triggered]
    material_changed = any('material-dropdown' in ti for ti in triggered_inputs)
    params_changed   = any('param-store' in ti for ti in triggered_inputs)
    switch_on = any('png-svg-switch' in ti for ti in triggered_inputs)

    if switch_on: return phi_fig, theta_fig, yz_fig
        
    p = SimParams(**store[material])
    
    m_array   = p.m_scale * (m_array_1 if material == '1' else m_array_2)
    M_array   = p.M_scale * (M_array_1 if material == '1' else M_array_2)
    K_array   = p.k_scale * (K_array_1 if material == '1' else K_array_2)
    alpha     = p.alpha_scale * (alpha_1 if material=='1' else alpha_2)
    lam       = p.lam_scale * (lam_1 if material=='1' else lam_2)
  
    h_index = np.abs(H_vals - H).argmin()
    T_vals  = T_vals_1 if material=='1' else T_vals_2
    t_index = np.abs(T_vals - T).argmin()

    theor_freqs_GHz = sorted(np.round([float(x[0, 0]) for (x, _) in compute_frequencies(H, m_array[t_index], M_array[t_index], K_array[t_index], gamma, alpha, lam)], 1), reverse=True)

    m_val   = m_array[t_index]
    M_val   = M_array[t_index]
    K_val   = K_array[t_index]
    kappa   = m_val / gamma
    sim_time, sol = run_simulation(H, m_val, M_val, K_val, alpha, lam, kappa, simulation_time=1e-9)
    time_ns = sim_time * 1e9
    theta   = np.degrees(sol[0])
    phi     = np.degrees(sol[1])
    
    # Выполнение аппроксимации
    if False:
        A1_theta = np.max(theta) / 2
        A2_theta = A1_theta
        A1_phi   = np.max(phi) / 2
        A2_phi   = A1_phi
    
        initial_guess_stage1 = [0, 2, 0, 2, 0, 2, 0, 2, theor_freqs_GHz[0], theor_freqs_GHz[1]]
        lower_bounds_stage1  = [-np.pi, 0.01, -np.pi, 0.01, -np.pi, 0.01, -np.pi, 0.01, 0.1, 0.1]
        upper_bounds_stage1  = [np.pi, 100, np.pi, 100, np.pi, 100, np.pi, 100, 120, 120]
    
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
    
    phi_fig   = create_phi_fig(time_ns, phi, phi_fit, H, T, approx_freqs_GHz, theor_freqs_GHz, material)
    theta_fig = create_theta_fig(time_ns, theta, theta_fit)
    yz_fig    = create_yz_fig(np.sin(np.pi/2 + np.radians(theta)) * np.sin(np.radians(phi)),
                                np.cos(np.pi/2 + np.radians(theta)),
                                time_ns, H)

    return phi_fig, theta_fig, yz_fig

@app.callback(
    Output('download-H-file', 'data'),
    Input('download-H-btn',  'n_clicks'),
    State('H-slider',        'value'),
    State('param-store',       'data'),
    State('material-dropdown', 'value'),
    prevent_initial_call=True,
)
def download_hfix(n_clicks, H, store, material):
    if not n_clicks:
        raise PreventUpdate
    def _make_hfix_npy(H, p, material):
        """Возвращает bytes содержимого .npy для фиксированного H."""
        T_vals  = T_vals_1 if material == '1' else T_vals_2
        m_vec   = p.m_scale * (m_array_1 if material == '1' else m_array_2)
        M_vec   = p.M_scale * (M_array_1 if material == '1' else M_array_2)
        K_vec   = p.k_scale * (K_array_1 if material == '1' else K_array_2)
        alpha   = p.alpha_scale * (alpha_1 if material == '1' else alpha_2)
        lam     = p.lam_scale * (lam_1 if material=='1' else lam_2)
    
        (f1, t1), (f2, t2) = compute_frequencies_H_fix(H, m_vec, M_vec, K_vec, gamma, alpha, lam)
        arr = np.vstack([T_vals, f1, f2, t1, t2])           # shape (3, N)
        buf = io.BytesIO()
        np.save(buf, arr); buf.seek(0)
        return buf.getvalue()
    p = SimParams(**store[material])
    content = _make_hfix_npy(H, p, material)
    return dcc.send_bytes(content, filename=f'H_{H/10:.0f}.npy')


@app.callback(
    Output('download-T-file',  'data'),
    Input('download-T-btn',    'n_clicks'),
    State('T-slider',          'value'),
    State('param-store',       'data'),
    State('material-dropdown', 'value'),
    prevent_initial_call=True,
)
def download_tfix(n_clicks, T, store, material):
    if not n_clicks:
        raise PreventUpdate
    def _make_tfix_npy(T, p, material):
        """Возвращает bytes содержимого .npy для фиксированной T."""
        H_vec = H_vals
        t_idx   = np.abs((T_vals_1 if material == '1' else T_vals_2) - T).argmin()
        m_val   = p.m_scale * (m_array_1 if material == '1' else m_array_2)[t_idx]
        M_val   = p.M_scale * (M_array_1 if material == '1' else M_array_2)[t_idx]
        K_val   = p.k_scale * (K_array_1  if material == '1' else K_array_2)[t_idx]
        alpha   = p.alpha_scale * (alpha_1 if material == '1' else alpha_2)
        lam     = p.lam_scale * (lam_1 if material=='1' else lam_2)
    
        (f1, t1), (f2, t2) = compute_frequencies_T_fix(H_vec, m_val, M_val, K_val, gamma, alpha, lam)
        arr = np.vstack([H_vec/10, f1, f2, t1, t2])            # Η (mT), shape (3, N)
        buf = io.BytesIO()
        np.save(buf, arr); buf.seek(0)
        return buf.getvalue()
    p = SimParams(**store[material])
    content = _make_tfix_npy(T, p, material)
    return dcc.send_bytes(content, filename=f'T_{T:.0f}.npy')

@app.callback(
    [Output('phase-graph', 'config'),
     Output('frequency-surface-graph', 'config'),
     Output('H_fix-graph', 'config'),
     Output('T_fix-graph', 'config'),
     Output('phi-graph', 'config'),
     Output('theta-graph', 'config'),
     Output('yz-graph', 'config')],
    Input('png-svg-switch', 'on'),
    prevent_initial_call=True,
)
def update_graph_config(svg_on):
    format = 'svg' if svg_on else 'png'
    phase_conf = {'toImageButtonOptions': {'format': format,'filename': 'ФД','width': w_phase, 'height': h_phase,'scale': 1 if format == 'svg' else 2}}
    freq_conf = {'toImageButtonOptions': {'format': format,'filename': 'Частоты 3D','width': w_freq, 'height': h_freq,'scale': 1 if format == 'svg' else 5}}
    H_fix_conf = {'toImageButtonOptions': {'format': format,'filename': 'H_fix','width': w_fix, 'height': h_fix,'scale': 1 if format == 'svg' else 2}}
    T_fix_conf = {'toImageButtonOptions': {'format': format,'filename': 'T_fix','width': w_fix, 'height': h_fix,'scale': 1 if format == 'svg' else 2}}
    phi_conf = {'toImageButtonOptions': {'format': format,'filename': 'Динамика phi','width': w_angles,'height': h_angles,'scale': 1 if format == 'svg' else 2}}
    theta_conf = {'toImageButtonOptions': {'format': format,'filename': 'Динамика theta','width': w_angles,'height': h_angles,'scale': 1 if format == 'svg' else 2}}
    yz_conf = {'toImageButtonOptions': {'format': format,'filename': 'Проекция траектории','width': w_yz, 'height': h_yz,'scale': 1 if format == 'svg' else 3}}

    return phase_conf, freq_conf, H_fix_conf, T_fix_conf, phi_conf, theta_conf, yz_conf

if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=8000)
