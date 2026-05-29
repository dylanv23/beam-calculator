"""
Professional Beam Analysis Tool — Streamlit App
Run with: streamlit run beam_analyzer.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beam Analysis Tool",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ────────────────────────────────────────────────────────── */
.stApp {
    background-color: #0f1117;
    color: #e2e8f0;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* ── Section headers ───────────────────────────────────────────────────── */
.section-title {
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #64748b;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2535;
}

/* ── Metric cards ──────────────────────────────────────────────────────── */
.metric-card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-unit {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* ── Info / reaction table ─────────────────────────────────────────────── */
.reaction-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}
.reaction-table th {
    background: #1e2535;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.5rem 0.8rem;
    text-align: left;
}
.reaction-table td {
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #1e2535;
    color: #e2e8f0;
}
.reaction-table tr:last-child td {
    border-bottom: none;
}

/* ── Tag badge ─────────────────────────────────────────────────────────── */
.tag {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
}
.tag-pin   { background:#1e3a5f; color:#60a5fa; }
.tag-roller{ background:#1e3d2f; color:#4ade80; }
.tag-fixed { background:#3d1e2f; color:#f472b6; }

/* ── Alert / warning ───────────────────────────────────────────────────── */
.warn-box {
    background: #2d1e0f;
    border: 1px solid #92400e;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-top: 0.5rem;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    border: 1px solid #1e2535;
    transition: all 0.15s ease;
}

/* Analyse (primary) */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: #fff;
    border: none;
    padding: 0.7rem 2rem;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    box-shadow: 0 0 16px rgba(59,130,246,0.4);
}

/* ── Inputs ────────────────────────────────────────────────────────────── */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: #1e2535 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-size: 0.88rem !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
details summary {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.04em;
}

/* ── Dividers ──────────────────────────────────────────────────────────── */
hr {
    border-color: #1e2535;
    margin: 0.8rem 0;
}

/* ── Page title area ───────────────────────────────────────────────────── */
.page-header {
    margin-bottom: 1.5rem;
}
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.page-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ── Status dot ────────────────────────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-ok  { background:#22c55e; }
.dot-err { background:#ef4444; }
.dot-warn{ background:#f59e0b; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION-STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
def _default(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_default("unit_system", "SI")          # SI | Imperial
_default("supports", [])               # list of {loc, type}
_default("point_loads", [])            # list of {loc, mag}
_default("dist_loads", [])             # list of {start, end, w_start, w_end}
_default("results", None)
_default("fig_beam", None)
_default("fig_diagrams", None)


# ─────────────────────────────────────────────────────────────────────────────
# UNIT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
US = st.session_state["unit_system"]

UNITS = {
    "SI": {
        "length": "m",  "force": "kN",  "moment": "kN·m",
        "stress": "MPa","deflection": "mm", "E": "GPa",
        "I": "cm⁴",     "dist_load": "kN/m",
        "depth": "mm",
    },
    "Imperial": {
        "length": "ft", "force": "kip", "moment": "kip·ft",
        "stress": "ksi","deflection": "in", "E": "ksi",
        "I": "in⁴",     "dist_load": "kip/ft",
        "depth": "in",
    },
}

def u(key):
    return UNITS[st.session_state["unit_system"]][key]


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def to_si_length(val):
    if st.session_state["unit_system"] == "Imperial":
        return val * 0.3048
    return val

def to_si_force(val):
    if st.session_state["unit_system"] == "SI":
        return val * 1e3          # kN → N
    else:
        return val * 4448.22      # kip → N

def to_si_E(val):
    if st.session_state["unit_system"] == "SI":
        return val * 1e9          # GPa → Pa
    else:
        return val * 6894757.29   # ksi → Pa

def to_si_I(val):
    if st.session_state["unit_system"] == "SI":
        return val * 1e-8         # cm⁴ → m⁴
    else:
        return val * 4.16231e-7   # in⁴ → m⁴

def to_si_depth(val):
    if st.session_state["unit_system"] == "SI":
        return val * 1e-2         # cm → m
    else:
        return val * 0.0254       # in → m

def to_si_dist(val):
    if st.session_state["unit_system"] == "SI":
        return val * 1e3
    else:
        return val * 14593.9


def analyse_beam(L_in, E_in, I_in, depth_in, supports, point_loads, dist_loads):
    """
    Finite Element Method (1D Beam Elements) analysis.
    Returns dict with arrays + scalar summary values.
    """
    n = 500
    L = to_si_length(L_in)
    E = to_si_E(E_in)
    I = to_si_I(I_in)
    c = to_si_depth(depth_in) / 2.0   

    if L <= 0:
        return None

    xs = np.linspace(0, L, n + 1)
    dx = L / n

    EI = E * I
    dof = 2 * (n + 1)
    
    # Global Stiffness Matrix and Force Vector
    K = np.zeros((dof, dof))
    F = np.zeros(dof)

    # Assemble K (1D Euler-Bernoulli Beam Element)
    L_e = dx
    k_e = (EI / L_e**3) * np.array([
        [12, 6*L_e, -12, 6*L_e],
        [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
        [-12, -6*L_e, 12, -6*L_e],
        [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
    ])

    for i in range(n):
        idx = 2 * i
        K[idx:idx+4, idx:idx+4] += k_e

    # Convert point loads 
    for pl in point_loads:
        loc = to_si_length(pl["loc"])
        mag = to_si_force(pl["mag"])
        idx = int(round(np.clip(loc / L, 0, 1) * n))
        F[2 * idx] += mag

    # Convert distributed loads (Consistent Nodal Forces)
    for dl in dist_loads:
        x0 = to_si_length(dl["start"])
        x1 = to_si_length(dl["end"])
        w0 = to_si_dist(dl["w_start"])
        w1 = to_si_dist(dl["w_end"])
        
        if x0 > x1:
            x0, x1 = x1, x0
            w0, w1 = w1, w0
        if x0 == x1: continue

        for i in range(n):
            xA = i * dx
            xB = (i + 1) * dx
            oA = max(x0, xA)
            oB = min(x1, xB)
            
            if oA < oB:
                qA = w0 + (w1 - w0) * (oA - x0) / (x1 - x0)
                qB = w0 + (w1 - w0) * (oB - x0) / (x1 - x0)
                q_avg = (qA + qB) / 2.0
                L_ov = oB - oA
                
                # Apply load proportionately to nodes (adequate resolution with n=500)
                F_y = q_avg * L_ov / 2.0
                M_z = q_avg * L_ov**2 / 12.0
                F[2*i] += F_y
                F[2*i+1] += M_z
                F[2*(i+1)] += F_y
                F[2*(i+1)+1] -= M_z

    # Apply support boundary conditions
    fixed_dofs = []
    supp_si = []
    
    for s in supports:
        loc = to_si_length(s["loc"])
        supp_si.append({"loc": loc, "type": s["type"]})
        idx = int(round(np.clip(loc / L, 0, 1) * n))
        
        if s["type"] in ("Pin", "Roller"):
            fixed_dofs.append(2 * idx)
        elif s["type"] == "Fixed":
            fixed_dofs.extend([2 * idx, 2 * idx + 1])

    fixed_dofs = list(set(fixed_dofs))

    if len(fixed_dofs) < 2 and not any(s["type"] == "Fixed" for s in supports):
        return {"error": "Beam is under-constrained. Add at least two supports or one fixed support."}

    # Solve for displacement 
    free_dofs = np.setdiff1d(np.arange(dof), fixed_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    try:
        U_f = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError:
        return {"error": "Singular stiffness matrix — check support configuration."}

    U = np.zeros(dof)
    U[free_dofs] = U_f

    w = U[0::2]
    
    # Calculate Support Reactions
    R = K.dot(U) - F 
    reactions = {}
    for s in supp_si:
        idx = int(round(np.clip(s["loc"] / L, 0, 1) * n))
        reactions[s["loc"]] = {
            "type": s["type"],
            "Ry": float(R[2 * idx]), 
        }

    # Derive Internal Forces (Shear & Bending Moment)
    M = np.zeros(n + 1)
    V = np.zeros(n + 1)
    
    for i in range(n):
        u_e = U[2*i : 2*i+4]
        M[i] = EI * (-6/dx**2 * u_e[0] - 4/dx * u_e[1] + 6/dx**2 * u_e[2] - 2/dx * u_e[3])
        V[i] = EI * (12/dx**3 * u_e[0] + 6/dx**2 * u_e[1] - 12/dx**3 * u_e[2] + 6/dx**2 * u_e[3])

    # End point assignment 
    u_e = U[2*(n-1) : 2*(n-1)+4]
    M[n] = EI * (6/dx**2 * u_e[0] + 2/dx * u_e[1] - 6/dx**2 * u_e[2] + 4/dx * u_e[3])
    V[n] = EI * (12/dx**3 * u_e[0] + 6/dx**2 * u_e[1] - 12/dx**3 * u_e[2] + 6/dx**2 * u_e[3])

    # Bending stress σ = M·c / I
    sigma = M * c / I

    # Convert outputs to display units
    if st.session_state["unit_system"] == "SI":
        w_disp   = w * 1e3              
        V_disp   = V * 1e-3             
        M_disp   = M * 1e-3             
        s_disp   = sigma * 1e-6         
        xs_disp  = xs                   
        rxn_fac  = 1e-3                 
    else:
        w_disp   = w / 0.0254           
        V_disp   = V / 4448.22          
        M_disp   = M / (4448.22 * 0.3048)  
        s_disp   = sigma / 6894757.29   
        xs_disp  = xs / 0.3048          
        rxn_fac  = 1 / 4448.22          

    rxn_display = {}
    for loc_si, rv in reactions.items():
        loc_d = loc_si if st.session_state["unit_system"] == "SI" else loc_si / 0.3048
        rxn_display[round(loc_d, 4)] = {
            "type": rv["type"],
            "Ry": rv["Ry"] * rxn_fac,
        }

    return {
        "xs": xs_disp,
        "w":  w_disp,
        "V":  V_disp,
        "M":  M_disp,
        "sigma": s_disp,
        "max_V":   float(np.max(np.abs(V_disp))),
        "max_M":   float(np.max(np.abs(M_disp))),
        "max_w":   float(np.max(np.abs(w_disp))),
        "max_sig": float(np.max(np.abs(s_disp))),
        "reactions": rxn_display,
        "L_disp": float(xs_disp[-1]),
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BEAM VISUALISATION (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG   = "#0f1117"
PLOT_PAPER= "#0f1117"
GRID_COL  = "#1e2535"
AXIS_COL  = "#4a5568"
TEXT_COL  = "#94a3b8"
LINE_BLUE = "#3b82f6"
LINE_RED  = "#ef4444"
LINE_GRN  = "#22c55e"
LINE_YEL  = "#f59e0b"
LINE_PUR  = "#a78bfa"

def _plotly_layout(title="", xlab="", ylab=""):
    return dict(
        title=dict(text=title, font=dict(size=13, color=TEXT_COL), x=0.0, xanchor='left'),
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", size=11, color=TEXT_COL),
        xaxis=dict(
            title=xlab, gridcolor=GRID_COL, zerolinecolor=AXIS_COL,
            tickfont=dict(size=10), title_font=dict(size=11),
            showgrid=True, zeroline=True,
        ),
        yaxis=dict(
            title=ylab, gridcolor=GRID_COL, zerolinecolor=AXIS_COL,
            tickfont=dict(size=10), title_font=dict(size=11),
            showgrid=True, zeroline=True,
        ),
        margin=dict(l=55, r=20, t=38, b=45),
        hoverlabel=dict(bgcolor="#1e2535", font_size=11, font_family="Inter"),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=TEXT_COL, activecolor=LINE_BLUE),
    )


def make_beam_figure(L_in, supports, point_loads, dist_loads):
    """Schematic beam diagram (not deflected)."""
    L = float(L_in)
    fig = go.Figure()

    # ── Beam body ────────────────────────────────────────────────────────────
    fig.add_shape(type="rect",
        x0=0, x1=L, y0=-0.12, y1=0.12,
        fillcolor="#2d3748", line=dict(color="#4a5568", width=1.2),
        layer="below"
    )

    # ── Neutral axis ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0, L], y=[0, 0],
        mode="lines",
        line=dict(color="#4a5568", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Supports ─────────────────────────────────────────────────────────────
    for s in supports:
        x  = float(s["loc"])
        tp = s["type"]
        col = {"Pin": LINE_BLUE, "Roller": LINE_GRN, "Fixed": LINE_PUR}.get(tp, TEXT_COL)

        if tp == "Pin":
            # Triangle
            tri_x = [x - 0.18, x, x + 0.18, x - 0.18]
            tri_y = [-0.12, -0.42, -0.12, -0.12]
            fig.add_trace(go.Scatter(
                x=tri_x, y=tri_y, mode="lines",
                fill="toself", fillcolor=col,
                line=dict(color=col, width=1.5),
                showlegend=False, hovertemplate=f"Pin @ {x} {u('length')}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[x - 0.22, x + 0.22], y=[-0.46, -0.46],
                mode="lines", line=dict(color=col, width=2),
                showlegend=False, hoverinfo="skip",
            ))

        elif tp == "Roller":
            # Triangle + circle
            tri_x = [x - 0.18, x, x + 0.18, x - 0.18]
            tri_y = [-0.12, -0.36, -0.12, -0.12]
            fig.add_trace(go.Scatter(
                x=tri_x, y=tri_y, mode="lines",
                fill="toself", fillcolor=col,
                line=dict(color=col, width=1.5),
                showlegend=False, hovertemplate=f"Roller @ {x} {u('length')}<extra></extra>",
            ))
            theta = np.linspace(0, 2 * np.pi, 30)
            fig.add_trace(go.Scatter(
                x=x + 0.07 * np.cos(theta), y=-0.44 + 0.07 * np.sin(theta),
                mode="lines", fill="toself", fillcolor=col,
                line=dict(color=col, width=1),
                showlegend=False, hoverinfo="skip",
            ))

        elif tp == "Fixed":
            fig.add_shape(type="rect",
                x0=x - 0.06, x1=x + 0.06, y0=-0.12, y1=0.12,
                fillcolor=col, line=dict(color=col, width=0),
                layer="above"
            )
            for dy in np.linspace(-0.10, 0.10, 5):
                fig.add_shape(type="line",
                    x0=x + 0.06, x1=x + 0.16, y0=dy, y1=dy - 0.06,
                    line=dict(color=col, width=1.5),
                )

    # ── Point loads ──────────────────────────────────────────────────────────
    for pl in point_loads:
        x   = float(pl["loc"])
        mag = float(pl["mag"])
        col = LINE_RED if mag < 0 else LINE_YEL
        ay = 0.12 if mag >= 0 else -0.12
        ey = 0.65 if mag >= 0 else -0.65
        fig.add_annotation(
            x=x, y=ay, ax=x, ay=ey,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.3, arrowwidth=2.2,
            arrowcolor=col,
        )
        fig.add_trace(go.Scatter(
            x=[x], y=[ey + (0.08 if mag >= 0 else -0.08)],
            mode="text",
            text=[f"{abs(mag):.1f} {u('force')}"],
            textfont=dict(size=10, color=col),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Distributed loads ────────────────────────────────────────────────────
    for i, dl in enumerate(dist_loads):
        x0 = float(dl["start"]); x1 = float(dl["end"])
        w0 = float(dl["w_start"]); w1 = float(dl["w_end"])
        col = "#f97316"
        n_arr = max(4, int((x1 - x0) / (L / 20)) + 1)
        xs_arr = np.linspace(x0, x1, n_arr)
        ws_arr = np.interp(xs_arr, [x0, x1], [w0, w1])
        top_y_arr = []
        for xi, wi in zip(xs_arr, ws_arr):
            mag_norm = 0.5 * abs(wi) / (max(abs(w0), abs(w1), 1e-9))
            ay_val = 0.12 if wi < 0 else -0.12
            ey_val = ay_val + (0.55 if wi >= 0 else -0.55) * max(mag_norm, 0.2)
            fig.add_annotation(
                x=xi, y=ay_val, ax=xi, ay=ey_val,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=0.9,
                arrowwidth=1.5, arrowcolor=col,
            )
            top_y_arr.append(ey_val)
        fig.add_trace(go.Scatter(
            x=list(xs_arr) + [xs_arr[-1], xs_arr[0]],
            y=top_y_arr + [0.12 if w1 < 0 else -0.12, 0.12 if w0 < 0 else -0.12],
            mode="lines", fill="toself",
            fillcolor="rgba(249,115,22,0.12)",
            line=dict(color=col, width=1.5),
            showlegend=False,
            hovertemplate=f"Dist. load #{i+1}<extra></extra>",
        ))

    # ── Dimension line ───────────────────────────────────────────────────────
    fig.add_annotation(
        x=0, y=-0.82, ax=L, ay=-0.82,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor=TEXT_COL,
    )
    fig.add_annotation(
        x=L, y=-0.82, ax=0, ay=-0.82,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor=TEXT_COL,
    )
    fig.add_trace(go.Scatter(
        x=[L / 2], y=[-0.96],
        mode="text",
        text=[f"L = {L:.2f} {u('length')}"],
        textfont=dict(size=11, color=TEXT_COL),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Layout ───────────────────────────────────────────────────────────────
    pad = L * 0.10 + 0.4
    layout = _plotly_layout("Beam Schematic", f"Position ({u('length')})", "")
    layout["yaxis"]["visible"]   = False
    layout["yaxis"]["range"]     = [-1.1, 1.1]
    layout["xaxis"]["range"]     = [-pad, L + pad]
    layout["height"]             = 260
    layout["margin"]             = dict(l=20, r=20, t=38, b=35)
    fig.update_layout(**layout)
    return fig


def make_diagram_figures(res):
    xs = res["xs"]
    L  = res["L_disp"]

    def fill_trace(xs, ys, col, name):
        return go.Scatter(
            x=xs, y=ys, mode="lines",
            fill="tozeroy", fillcolor=col.replace(")", ",0.18)").replace("rgb","rgba"),
            line=dict(color=col, width=2),
            name=name,
            hovertemplate=f"%{{x:.3f}} {u('length')}<br>%{{y:.4f}}<extra>{name}</extra>",
        )

    # ── Shear ────────────────────────────────────────────────────────────────
    fig_V = go.Figure()
    fig_V.add_trace(fill_trace(xs, res["V"], LINE_BLUE, f"Shear ({u('force')})"))
    fig_V.add_hline(y=0, line=dict(color=AXIS_COL, width=1))
    fig_V.update_layout(**_plotly_layout(
        "Shear Force Diagram",
        f"Position ({u('length')})",
        f"Shear ({u('force')})",
    ))

    # ── Moment ───────────────────────────────────────────────────────────────
    fig_M = go.Figure()
    fig_M.add_trace(fill_trace(xs, res["M"], LINE_RED, f"Moment ({u('moment')})"))
    fig_M.add_hline(y=0, line=dict(color=AXIS_COL, width=1))
    fig_M.update_layout(**_plotly_layout(
        "Bending Moment Diagram",
        f"Position ({u('length')})",
        f"Moment ({u('moment')})",
    ))

    # ── Deflection ───────────────────────────────────────────────────────────
    fig_w = go.Figure()
    fig_w.add_trace(fill_trace(xs, res["w"], LINE_GRN, f"Deflection ({u('deflection')})"))
    fig_w.add_hline(y=0, line=dict(color=AXIS_COL, width=1))
    fig_w.update_layout(**_plotly_layout(
        "Deflection Diagram",
        f"Position ({u('length')})",
        f"Deflection ({u('deflection')})",
    ))

    # ── Bending stress ───────────────────────────────────────────────────────
    fig_s = go.Figure()
    fig_s.add_trace(fill_trace(xs, res["sigma"], LINE_PUR, f"Stress ({u('stress')})"))
    fig_s.add_hline(y=0, line=dict(color=AXIS_COL, width=1))
    fig_s.update_layout(**_plotly_layout(
        "Bending Stress Diagram",
        f"Position ({u('length')})",
        f"Stress ({u('stress')})",
    ))

    return fig_V, fig_M, fig_w, fig_s


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT PANEL
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Unit toggle ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Unit System</div>', unsafe_allow_html=True)
    unit_col1, unit_col2 = st.columns(2)
    with unit_col1:
        if st.button("SI (kN, m)", use_container_width=True,
                     type="primary" if st.session_state["unit_system"] == "SI" else "secondary"):
            st.session_state["unit_system"] = "SI"
            st.session_state["results"] = None
            st.rerun()
    with unit_col2:
        if st.button("Imperial (kip, ft)", use_container_width=True,
                     type="primary" if st.session_state["unit_system"] == "Imperial" else "secondary"):
            st.session_state["unit_system"] = "Imperial"
            st.session_state["results"] = None
            st.rerun()

    st.markdown("---")

    # ── Beam properties ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Beam Properties</div>', unsafe_allow_html=True)

    L_val = st.number_input(f"Length ({u('length')})", min_value=0.1, max_value=1e6,
                            value=10.0, step=0.5, format="%.2f")

    col_E, col_I = st.columns(2)
    with col_E:
        E_val = st.number_input(f"E ({u('E')})", min_value=0.001,
                                value=200.0 if st.session_state["unit_system"] == "SI" else 29000.0,
                                step=1.0, format="%.1f")
    with col_I:
        I_val = st.number_input(f"I ({u('I')})", min_value=0.001,
                                value=1000.0 if st.session_state["unit_system"] == "SI" else 240.0,
                                step=1.0, format="%.2f")

    depth_val = st.number_input(f"Section Depth ({u('depth')})", min_value=0.1,
                                value=200.0 if st.session_state["unit_system"] == "SI" else 8.0,
                                step=1.0, format="%.1f")

    st.markdown("---")

    # ── Supports ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Supports</div>', unsafe_allow_html=True)

    with st.expander("Add / Remove Supports", expanded=True):
        c1, c2 = st.columns([2, 2])
        with c1:
            s_loc = st.number_input(f"Position ({u('length')})", min_value=0.0,
                                    max_value=float(L_val), value=0.0,
                                    step=0.5, format="%.2f", key="s_loc")
        with c2:
            s_type = st.selectbox("Type", ["Pin", "Roller", "Fixed"], key="s_type")

        if st.button("Add Support", use_container_width=True):
            st.session_state["supports"].append({"loc": s_loc, "type": s_type})
            st.session_state["results"] = None

        if st.session_state["supports"]:
            df_s = pd.DataFrame(st.session_state["supports"])
            df_s.columns = [f"Position ({u('length')})", "Type"]
            st.dataframe(df_s, use_container_width=True, height=120,
                         hide_index=True)
            if st.button("Clear All Supports", use_container_width=True):
                st.session_state["supports"] = []
                st.session_state["results"] = None

    st.markdown("---")

    # ── Point loads ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Point Loads</div>', unsafe_allow_html=True)

    with st.expander("Add / Remove Point Loads", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            pl_loc = st.number_input(f"Position ({u('length')})", min_value=0.0,
                                     max_value=float(L_val), value=float(L_val) / 2,
                                     step=0.5, format="%.2f", key="pl_loc")
        with c2:
            pl_mag = st.number_input(f"Magnitude ({u('force')})",
                                     value=-10.0, step=1.0, format="%.2f", key="pl_mag")

        st.caption("⬇ Negative = downward")

        if st.button("Add Point Load", use_container_width=True):
            st.session_state["point_loads"].append({"loc": pl_loc, "mag": pl_mag})
            st.session_state["results"] = None

        if st.session_state["point_loads"]:
            df_pl = pd.DataFrame(st.session_state["point_loads"])
            df_pl.columns = [f"Position ({u('length')})", f"Magnitude ({u('force')})"]
            st.dataframe(df_pl, use_container_width=True, height=120, hide_index=True)
            if st.button("Clear Point Loads", use_container_width=True):
                st.session_state["point_loads"] = []
                st.session_state["results"] = None

    st.markdown("---")

    # ── Distributed loads ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Distributed Loads</div>', unsafe_allow_html=True)

    with st.expander("Add / Remove Distributed Loads", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            dl_x0 = st.number_input(f"Start ({u('length')})", min_value=0.0,
                                    max_value=float(L_val), value=0.0,
                                    step=0.5, format="%.2f", key="dl_x0")
        with c2:
            dl_x1 = st.number_input(f"End ({u('length')})", min_value=0.0,
                                    max_value=float(L_val), value=float(L_val),
                                    step=0.5, format="%.2f", key="dl_x1")

        c3, c4 = st.columns(2)
        with c3:
            dl_w0 = st.number_input(f"w₀ ({u('dist_load')})", value=-5.0,
                                    step=0.5, format="%.2f", key="dl_w0")
        with c4:
            dl_w1 = st.number_input(f"w₁ ({u('dist_load')})", value=-5.0,
                                    step=0.5, format="%.2f", key="dl_w1")

        st.caption("⬇ Negative = downward. Equal w₀/w₁ = UDL, different = trapezoidal.")

        if st.button("Add Distributed Load", use_container_width=True):
            st.session_state["dist_loads"].append({
                "start": dl_x0, "end": dl_x1,
                "w_start": dl_w0, "w_end": dl_w1,
            })
            st.session_state["results"] = None

        if st.session_state["dist_loads"]:
            df_dl = pd.DataFrame(st.session_state["dist_loads"])
            df_dl.columns = [
                f"Start ({u('length')})", f"End ({u('length')})",
                f"w₀ ({u('dist_load')})", f"w₁ ({u('dist_load')})"
            ]
            st.dataframe(df_dl, use_container_width=True, height=120, hide_index=True)
            if st.button("Clear Distributed Loads", use_container_width=True):
                st.session_state["dist_loads"] = []
                st.session_state["results"] = None

    st.markdown("---")

    # ── Analyse button ───────────────────────────────────────────────────────
    run_analysis = st.button("▶  Run Analysis", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-title">🏗️ Beam Analysis Tool</div>
  <div class="page-subtitle">
    Configure beam dimensions, supports, and loads — then run a structural analysis.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Beam schematic ────────────────────────────────────────────────────────────
beam_fig = make_beam_figure(
    L_val,
    st.session_state["supports"],
    st.session_state["point_loads"],
    st.session_state["dist_loads"],
)
st.plotly_chart(beam_fig, use_container_width=True, config={
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["toImage"],
    "toImageButtonOptions": {"format": "png", "filename": "beam_schematic", "scale": 2},
    "scrollZoom": True,
})

st.markdown("---")

# ── Run analysis ──────────────────────────────────────────────────────────────
if run_analysis:
    if len(st.session_state["supports"]) < 2 and not any(s["type"] == "Fixed" for s in st.session_state["supports"]):
        st.markdown('<div class="warn-box">⚠ Add at least two supports (or one fixed support) before running the analysis.</div>',
                    unsafe_allow_html=True)
    elif not st.session_state["point_loads"] and not st.session_state["dist_loads"]:
        st.markdown('<div class="warn-box">⚠ Add at least one load before running the analysis.</div>',
                    unsafe_allow_html=True)
    else:
        with st.spinner("Solving…"):
            res = analyse_beam(
                L_val, E_val, I_val, depth_val,
                st.session_state["supports"],
                st.session_state["point_loads"],
                st.session_state["dist_loads"],
            )
        if res and res.get("error"):
            st.markdown(f'<div class="warn-box">⚠ {res["error"]}</div>', unsafe_allow_html=True)
            st.session_state["results"] = None
        else:
            st.session_state["results"] = res

# ── Results display ────────────────────────────────────────────────────────────
res = st.session_state.get("results")

if res:
    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("### Results Summary")
    m1, m2, m3, m4 = st.columns(4)

    def metric_card(col, label, value, unit):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value:.4g}</div>
              <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    metric_card(m1, "Max Shear",        res["max_V"],   u("force"))
    metric_card(m2, "Max Moment",       res["max_M"],   u("moment"))
    metric_card(m3, "Max Deflection",   res["max_w"],   u("deflection"))
    metric_card(m4, "Max Bending Stress", res["max_sig"], u("stress"))

    st.markdown("---")

    # ── Support reactions ─────────────────────────────────────────────────────
    st.markdown("### Support Reactions")
    rxn_rows = []
    for loc, rv in res["reactions"].items():
        rxn_rows.append({
            f"Position ({u('length')})": round(loc, 4),
            "Type": rv["type"],
            f"Vertical Reaction ({u('force')})": round(rv["Ry"], 6),
        })
    if rxn_rows:
        df_rxn = pd.DataFrame(rxn_rows)
        st.dataframe(df_rxn, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Diagrams ──────────────────────────────────────────────────────────────
    st.markdown("### Diagrams")
    st.caption("Use the toolbar to zoom, pan, and download each diagram as PNG.")

    fig_V, fig_M, fig_w, fig_s = make_diagram_figures(res)

    CHART_CFG = {
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["toImage"],
        "toImageButtonOptions": {"format": "png", "scale": 2},
        "scrollZoom": True,
        "displaylogo": False,
    }

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Shear Force", "📈 Bending Moment", "📉 Deflection", "🔴 Bending Stress"
    ])

    with tab1:
        fig_V.update_layout(height=380)
        st.plotly_chart(fig_V, use_container_width=True, config={
            **CHART_CFG,
            "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "shear_force_diagram"},
        })

    with tab2:
        fig_M.update_layout(height=380)
        st.plotly_chart(fig_M, use_container_width=True, config={
            **CHART_CFG,
            "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "bending_moment_diagram"},
        })

    with tab3:
        fig_w.update_layout(height=380)
        st.plotly_chart(fig_w, use_container_width=True, config={
            **CHART_CFG,
            "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "deflection_diagram"},
        })

    with tab4:
        fig_s.update_layout(height=380)
        st.plotly_chart(fig_s, use_container_width=True, config={
            **CHART_CFG,
            "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "bending_stress_diagram"},
        })

    # ── All-in-one combined export ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Export — Combined Report Chart")
    if st.button("📥  Generate Combined PNG (all 4 diagrams)"):
        fig_all = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Shear Force", "Bending Moment", "Deflection", "Bending Stress"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        def add_to_subplot(fig_src, row, col):
            for trace in fig_src.data:
                fig_all.add_trace(trace, row=row, col=col)

        add_to_subplot(fig_V, 1, 1)
        add_to_subplot(fig_M, 1, 2)
        add_to_subplot(fig_w, 2, 1)
        add_to_subplot(fig_s, 2, 2)

        fig_all.update_layout(
            paper_bgcolor=PLOT_PAPER,
            plot_bgcolor=PLOT_BG,
            font=dict(family="Inter, sans-serif", size=11, color=TEXT_COL),
            height=700,
            showlegend=False,
            title=dict(
                text="Beam Analysis — Combined Diagrams",
                font=dict(size=14, color=TEXT_COL), x=0.5
            ),
        )
        fig_all.update_xaxes(gridcolor=GRID_COL, zerolinecolor=AXIS_COL)
        fig_all.update_yaxes(gridcolor=GRID_COL, zerolinecolor=AXIS_COL)

        img_bytes = fig_all.to_image(format="png", scale=2, width=1400, height=700)
        st.download_button(
            label="⬇  Download Combined PNG",
            data=img_bytes,
            file_name="beam_analysis_combined.png",
            mime="image/png",
        )
        st.plotly_chart(fig_all, use_container_width=True,
                        config={"displayModeBar": False})

else:
    # placeholder when not yet analysed
    st.markdown("""
    <div style="
        background:#161b27;
        border:1px solid #1e2535;
        border-radius:12px;
        padding:3rem 2rem;
        text-align:center;
        color:#4a5568;
        margin-top:1rem;
    ">
      <div style="font-size:2.5rem;margin-bottom:0.8rem;">📐</div>
      <div style="font-size:1rem;font-weight:600;color:#64748b;margin-bottom:0.4rem;">
        No results yet
      </div>
      <div style="font-size:0.83rem;">
        Configure your beam in the sidebar, then click <strong style="color:#3b82f6">Run Analysis</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:0.75rem;color:#2d3748;text-align:center;padding:0.5rem 0;">
  Beam Analysis Tool · Finite Element Method (1D Beam Elements) · 500 nodes
</div>
""", unsafe_allow_html=True)