from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# ─── paths ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "scored.csv"
MODEL_DIR   = ROOT / "artifacts" / "models"
W_JSON      = MODEL_DIR / "ensemble_weights.json"

# ─── helpers ───────────────────────────────────────────────────────────────
@st.cache_data
def load_scored(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"isFraud", "iso_score", "ae_score", "gmm_score"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must include {need}")
    return df

def precision_at_k(y: pd.Series, s: pd.Series, k_frac: float) -> float:
    top_n = max(1, int(len(s) * k_frac))
    idx   = np.argsort(s.values)[-top_n:]
    return precision_score(y.iloc[idx], np.ones(top_n, dtype=int))

# ─── page config ───────────────────────────────────────────────────────────
st.set_page_config("Precision-@-k% Dashboard", layout="wide")
st.markdown('<h1 style="font-size:1.6em">Precision-@-k% Dashboard</h1>', unsafe_allow_html=True)

# ─── sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("Data & Ensemble Weights")
csv_path = st.sidebar.text_input("Scored CSV path", str(DEFAULT_CSV))
df = load_scored(csv_path)

default_w = (0.4, 0.3, 0.3)
if W_JSON.exists():
    default_w = tuple(json.loads(W_JSON.read_text()).values())

w_iso = st.sidebar.slider("IF weight",  0.0, 1.0, default_w[0], 0.01, format="%.2f")
w_ae  = st.sidebar.slider("AE weight",  0.0, 1.0, default_w[1], 0.01, format="%.2f")
w_gmm = st.sidebar.slider("GMM weight", 0.0, 1.0, default_w[2], 0.01, format="%.2f")

if st.sidebar.checkbox("Normalise weights to 1", True):
    S = w_iso + w_ae + w_gmm
    if S:
        w_iso, w_ae, w_gmm = [w / S for w in (w_iso, w_ae, w_gmm)]

st.sidebar.header("Curves")
show_if  = st.sidebar.checkbox("Isolation Forest (IF)", True)
show_ae  = st.sidebar.checkbox("Autoencoder (AE)",      True)
show_gmm = st.sidebar.checkbox("Gaussian MM (GMM)",     True)
show_ens = st.sidebar.checkbox("Ensemble (ENS)",        True)

# ─── ensemble score ────────────────────────────────────────────────────────
df["ensemble_score"] = w_iso * df.iso_score + w_ae * df.ae_score + w_gmm * df.gmm_score

# ─── precision-k curve ─────────────────────────────────────────────────────
k_vals = np.arange(0.001, 1.001, 0.001)          # 0.1 … 100 % as fractions
y_true = df.isFraud.astype(int)

build = lambda col: [precision_at_k(y_true, df[col], k) for k in k_vals]
curves = {
    "IF":  build("iso_score"),
    "AE":  build("ae_score"),
    "GMM": build("gmm_score"),
    "ENS": build("ensemble_score"),
}

st.markdown('<h4 style="font-size:1.2em">Precision-@-k% Curve</h4>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(9, 3.5))  # static pixel-ish size
if show_if:  ax.plot(k_vals * 100, curves["IF"],  label="IF",  lw=1)
if show_ae:  ax.plot(k_vals * 100, curves["AE"],  label="AE",  lw=1)
if show_gmm: ax.plot(k_vals * 100, curves["GMM"], label="GMM", lw=1)
if show_ens: ax.plot(k_vals * 100, curves["ENS"], label="ENS", lw=2, color="black")
ax.set_xlim(0, 100)
ax.set_xlabel("Top k%")
ax.set_ylabel("Precision")
ax.legend()
st.pyplot(fig)

# ─── metrics table & lift block ────────────────────────────────────────────
outer_left, mid, outer_right = st.columns([1, 6, 1])
with mid:
    table_col, lift_col = st.columns([4, 1])

    # Key precision table
    table_col.markdown('<h4 style="font-size:1.1em">Key Precision Metrics</h4>', unsafe_allow_html=True)
    P = lambda col, k: precision_at_k(y_true, df[col], k)
    tbl = pd.DataFrame(
        {
            "IF":  [P("iso_score", .01), P("iso_score", .05), P("iso_score", .10)],
            "AE":  [P("ae_score",  .01), P("ae_score",  .05), P("ae_score",  .10)],
            "GMM": [P("gmm_score", .01), P("gmm_score", .05), P("gmm_score", .10)],
            "ENS": [P("ensemble_score", .01),
                    P("ensemble_score", .05),
                    P("ensemble_score", .10)],
        },
        index=["P@1 %", "P@5 %", "P@10 %"],
    ).round(6)
    html_tbl = (
        tbl.style
        .set_table_attributes('style="border-collapse:collapse;margin-left:0"')
        .set_table_styles([
            {"selector": "th, td",
             "props": [("border", "1px solid"),
                       ("padding", "10px 16px"),
                       ("text-align", "center"),
                       ("font-size", "1.05em")]}])
        .to_html()
    )
    table_col.markdown(html_tbl, unsafe_allow_html=True)

    # Lift section
    lift_col.markdown(
        '<h4 style="font-size:1.1em;white-space:nowrap;">'
        'Lift&nbsp;over&nbsp;Random&nbsp;@&nbsp;1&nbsp;%'
        '</h4>',
     unsafe_allow_html=True
    )
    ens_p1 = tbl.loc["P@1 %", "ENS"]
    baseline = y_true.mean()
    lift = ens_p1 / baseline if baseline else np.nan
    lift_html = (
        '<div style="display:flex;flex-direction:column;justify-content:center;'
        'align-items:flex-start;height:100%;">'
        f'<div style="font-size:3em;font-weight:bold;color:green;">'
        f'&#9650;&nbsp;{lift:.2f}×</div>'
        f'<div style="font-size:0.9em;">Baseline {baseline*100:.2f} %</div>'
        '</div>'
    )
    lift_col.markdown(lift_html, unsafe_allow_html=True)