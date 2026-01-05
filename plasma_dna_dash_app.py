import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar

rng = np.random.default_rng(42)

def P_transform(t_s, E_kVcm):
    alpha = 0.12
    return 1 - np.exp(-alpha * E_kVcm * t_s)

def P_viable(t_s, E_kVcm):
    beta = 0.05
    return np.exp(-beta * E_kVcm * t_s)

def temp_modifier(T_C, T_opt=25, sigma=7):
    return np.exp(-0.5 * ((T_C - T_opt) / sigma)**2)

def eta_effective(t, E, T):
    return P_transform(t, E) * P_viable(t, E) * temp_modifier(T) * 100

# Temperature-dependent DNA mobility
def dna_mobility(T_C, mu0=4e-8, T_ref=25):
    return mu0 * (1 + 0.01*(T_C - T_ref))

def monte_carlo_eta(t, E, T, runs=150):
    ts = np.clip(rng.normal(t, 0.4, runs), 0, None)
    Es = np.clip(rng.normal(E, 0.15, runs), 0, None)
    Ts = rng.normal(T, 1.0, runs)
    eta = eta_effective(ts, Es, Ts)
    return np.percentile(eta, [5, 50, 95])

def simulate_dna(E0, T_C, n_particles=80, steps=45):
    mu = dna_mobility(T_C)
    D = 1e-12 * (T_C / 25)
    dt = 0.05
    decay_len = 0.002

    x = rng.uniform(-0.004, 0.004, n_particles)
    y = rng.uniform(-0.004, 0.004, n_particles)
    z = np.zeros(n_particles)

    max_r = 0.006
    frames = []

    for _ in range(steps):
        r = np.sqrt(x**2 + y**2) + 1e-9
        E_local = E0 * np.exp(-r / decay_len) * 1e5

        # Drift + diffusion
        x += mu * E_local * dt + np.sqrt(2 * D * dt) * rng.normal(size=n_particles)
        y += mu * E_local * dt + np.sqrt(2 * D * dt) * rng.normal(size=n_particles)
        z += np.sqrt(2 * D * dt) * rng.normal(size=n_particles)

        max_r = max(max_r, np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)))

        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=x.copy(), y=y.copy(), z=z.copy(),
                mode='markers',
                marker=dict(
                    size=5,
                    color=E_local,
                    colorscale='Turbo',
                    cmin=0,
                    cmax=E0*1e5,
                    colorbar=dict(title='E (V/m)'),
                    opacity=0.8
                ),
                hovertemplate='x: %{x:.4f} m<br>y: %{y:.4f} m<br>z: %{z:.4f} m<br>E: %{marker.color:.1f} V/m'
            )]
        ))

    return frames, max_r

def find_optimal(T):
    res = minimize_scalar(lambda t: -eta_effective(t, 3, T), bounds=(0.1, 20), method='bounded')
    best_t = res.x
    best_E = 3
    best_eta = eta_effective(best_t, best_E, T)
    return best_eta, best_t, best_E

app = Dash(__name__)
app.title = "Helium Plasma DNA Simulator"

app.layout = html.Div([
    html.H2("Helium Plasma–Assisted DNA Transport", style={"textAlign": "center"}),

    html.Div([
        html.Label("Exposure Time (s)"),
        dcc.Slider(0.5, 20, 0.5, value=8, id="time"),

        html.Label("Electric Field (kV/cm)"),
        dcc.Slider(0.5, 8, 0.1, value=3, id="field"),

        html.Label("Temperature (°C)"),
        dcc.Slider(20, 45, 1, value=25, id="temp"),

        html.Br(),
        html.Button('Reset Sliders', id='reset-btn', n_clicks=0)
    ], style={"width": "80%", "margin": "auto"}),

    html.Br(),
    html.Div(id="optimal-text", style={"textAlign": "center", "fontSize": 18}),

    dcc.Graph(id="eta-uncertainty"),
    dcc.Graph(id="dna-animation")
])

@app.callback(
    Output("time", "value"),
    Output("field", "value"),
    Output("temp", "value"),
    Input('reset-btn', 'n_clicks')
)
def reset_sliders(n):
    return 8, 3, 25

@app.callback(
    Output("eta-uncertainty", "figure"),
    Output("dna-animation", "figure"),
    Output("optimal-text", "children"),
    Input("time", "value"),
    Input("field", "value"),
    Input("temp", "value")
)
def update(t, E, T):
    t_vals = np.linspace(0.5, 20, 40)
    p5, p50, p95 = np.array([monte_carlo_eta(tt, E, T) for tt in t_vals]).T

    fig_eta = go.Figure([ 
        go.Scatter(x=t_vals, y=p5, line=dict(width=0), showlegend=False),
        go.Scatter(x=t_vals, y=p95, fill='tonexty', fillcolor='rgba(0,120,200,0.25)', line=dict(width=0), name='90% CI'),
        go.Scatter(x=t_vals, y=p50, line=dict(color='blue'), name='Median Efficiency')
    ])
    fig_eta.update_layout(title='Transformation Efficiency (Helium Plasma)', xaxis_title='Time (s)', yaxis_title='Efficiency (%)')

    frames, axis_range = simulate_dna(E, T)
    fig_dna = go.Figure(data=frames[0].data, frames=frames)
    fig_dna.update_layout(
        title='DNA Motion Near Helium Plasma Electrode',
        scene=dict(
            xaxis=dict(range=[-axis_range, axis_range], title='x (m)'),
            yaxis=dict(range=[-axis_range, axis_range], title='y (m)'),
            zaxis=dict(range=[-axis_range, axis_range], title='z (m)')
        ),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{
                'label': '▶ Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 70, 'redraw': True}, 'fromcurrent': True}]
            }]
        }]
    )

    best_eta, best_t, best_E = find_optimal(T)
    text = f"Optimal efficiency at {T}°C ≈ {best_eta:.1f}% (Time ≈ {best_t:.1f}s, Field ≈ {best_E:.2f} kV/cm)"

    return fig_eta, fig_dna, text
if __name__ == "__main__":
    app.run(debug=False)
