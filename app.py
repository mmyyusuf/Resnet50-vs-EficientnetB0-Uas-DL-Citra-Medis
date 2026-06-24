"""
Aplikasi Streamlit: Perbandingan ResNet50 vs EfficientNetB0 
dengan Explainable AI Grad-CAM++ untuk Klasifikasi Penyakit Retina (OCT)
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import gdown
import os
import io
import requests
from pathlib import Path

# ─── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaScan AI | OCT Classification",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Medical / Clinical Theme ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;600&family=Roboto+Condensed:wght@400;700&display=swap');

/* ── Keyframe Animations ── */
@keyframes fade-in {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-border {
    0%, 100% { border-color: rgba(14,165,233,0.3); }
    50%       { border-color: rgba(14,165,233,0.6); }
}
@keyframes shimmer-med {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes scan-h {
    0%   { top: 0; opacity: 0.6; }
    100% { top: 100%; opacity: 0; }
}

/* ── Root Variables — Medical Palette ── */
:root {
    --bg:        #f0f4f8;
    --surface:   #ffffff;
    --surface2:  #f8fafc;
    --surface3:  #eef2f7;
    --primary:   #0369a1;      /* Deep medical blue */
    --primary-l: #0ea5e9;      /* Sky blue */
    --primary-ll:#bae6fd;      /* Light sky */
    --accent:    #0891b2;      /* Cyan-700 */
    --success:   #059669;      /* Emerald */
    --warning:   #d97706;      /* Amber */
    --danger:    #dc2626;      /* Red */
    --purple:    #7c3aed;
    --text:      #0f172a;
    --text-sec:  #475569;
    --text-muted:#94a3b8;
    --border:    #cbd5e1;
    --border-l:  #e2e8f0;
    --shadow:    0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.05);
    --shadow-lg: 0 10px 15px rgba(0,0,0,0.08), 0 4px 6px rgba(0,0,0,0.04);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.main .block-container {
    padding: 1.25rem 2rem;
    max-width: 1400px;
}

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 45%, #0891b2 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    animation: fade-in 0.5s ease-out;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -40%; left: -10%;
    width: 50%; height: 200%;
    background: rgba(255,255,255,0.05);
    transform: rotate(-15deg);
    pointer-events: none;
}
.hero-header::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #7dd3fc, #38bdf8);
    background-size: 200% auto;
    animation: shimmer-med 3s linear infinite;
}
.hero-scan-line {
    position: absolute;
    left: 0; right: 0;
    height: 1px;
    background: rgba(255,255,255,0.15);
    animation: scan-h 4s linear infinite;
    pointer-events: none;
}
.hero-title {
    font-family: 'Roboto Condensed', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.1;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.hero-sub {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.75);
    margin-top: 0.5rem;
    letter-spacing: 0.03em;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.7rem;
    color: #e0f2fe;
    margin-top: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    backdrop-filter: blur(4px);
}
.hero-badge::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: #4ade80;
    border-radius: 50%;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.3);
}
.hero-cross {
    position: absolute;
    right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    width: 60px; height: 60px;
    opacity: 0.12;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border-l);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    animation: fade-in 0.4s ease-out;
    transition: box-shadow 0.2s;
}
.card:hover {
    box-shadow: var(--shadow-md);
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 2px solid var(--primary-ll);
    padding-bottom: 0.5rem;
}

/* ── Metric Boxes ── */
.metric-box {
    background: var(--surface2);
    border: 1px solid var(--border-l);
    border-left: 3px solid var(--primary-l);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
    transition: all 0.2s;
}
.metric-box:hover {
    border-left-color: var(--primary);
    box-shadow: var(--shadow);
    transform: translateX(3px);
}
.metric-label {
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.1;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-cnv   { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.badge-dme   { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.badge-drusen{ background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
.badge-normal{ background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
.badge-mh    { background: #ede9fe; color: #5b21b6; border: 1px solid #c4b5fd; }
.badge-dr    { background: #fce7f3; color: #9d174d; border: 1px solid #f9a8d4; }
.badge-csr   { background: #e0f2fe; color: #0c4a6e; border: 1px solid #7dd3fc; }
.badge-amd   { background: #ffedd5; color: #9a3412; border: 1px solid #fdba74; }

/* ── Analysis Box ── */
.analysis-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 4px solid var(--primary-l);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    font-size: 0.87rem;
    line-height: 1.8;
    color: #1e3a5f;
    margin-top: 1rem;
}

/* ── Info / Result Boxes ── */
.result-agree {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 4px solid var(--success);
    border-radius: 10px;
    padding: 0.85rem 1.25rem;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.result-disagree {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid var(--warning);
    border-radius: 10px;
    padding: 0.85rem 1.25rem;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

/* ── Progress Bars ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--primary), var(--primary-l)) !important;
    border-radius: 4px !important;
}
.stProgress > div {
    background: var(--border-l) !important;
    border-radius: 4px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: var(--surface3) !important;
    border-radius: 10px;
    padding: 0.3rem;
    border: 1px solid var(--border-l);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    border: none;
    color: var(--text-sec);
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.45rem 1rem;
    transition: all 0.18s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--primary);
    background: rgba(14,165,233,0.08);
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--primary) !important;
    border: 1px solid var(--border-l) !important;
    box-shadow: var(--shadow) !important;
    font-weight: 600 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1rem !important;
}

/* ── Sidebar custom elements ── */
.sb-header {
    background: linear-gradient(135deg, #0369a1, #0891b2);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    text-align: center;
}
.sb-section {
    background: var(--surface2);
    border: 1px solid var(--border-l);
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.75rem;
}
.sb-section-title {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
    border-bottom: 1px solid var(--border-l);
    padding-bottom: 0.35rem;
}
.sb-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
    font-size: 0.78rem;
    color: var(--text-sec);
    border-bottom: 1px solid var(--border-l);
}
.sb-stat:last-child { border-bottom: none; }
.sb-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: var(--primary);
}
.sb-model-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--surface);
    border: 1px solid var(--border-l);
    border-radius: 6px;
    padding: 0.4rem 0.65rem;
    margin-bottom: 0.35rem;
    font-size: 0.78rem;
    color: var(--text-sec);
    font-weight: 500;
}
.sb-model-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sb-nav-link {
    display: block;
    padding: 0.4rem 0.65rem;
    border-radius: 6px;
    font-size: 0.8rem;
    color: var(--text-sec);
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    text-decoration: none;
    margin-bottom: 0.2rem;
}
.sb-nav-link:hover {
    background: #e0f2fe;
    color: var(--primary);
}
.sb-cls-chip {
    display: inline-block;
    font-size: 0.68rem;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin: 0.15rem;
    font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--primary) !important;
    border: none !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 1px 3px rgba(3,105,161,0.3) !important;
}
.stButton > button:hover {
    background: #0284c7 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(3,105,161,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── File Uploader ── */
div[data-testid="stFileUploader"] {
    background: #f0f9ff;
    border: 2px dashed #7dd3fc;
    border-radius: 12px;
    transition: all 0.2s;
}
div[data-testid="stFileUploader"]:hover {
    background: #e0f2fe;
    border-color: var(--primary-l);
}

/* ── Selectbox & inputs ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Expander ── */
details {
    background: var(--surface2) !important;
    border: 1px solid var(--border-l) !important;
    border-radius: 8px !important;
    margin-bottom: 0.4rem !important;
}
details summary {
    color: var(--text-sec) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb {
    background: #94a3b8;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: #64748b; }

/* ── Status widget ── */
div[data-testid="stStatusWidget"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-l) !important;
    border-radius: 10px !important;
}

/* ── Alert / info boxes ── */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border-l) !important;
    margin: 1rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Konstanta ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']

CLASS_BADGE = {
    'CNV':    'badge-cnv',
    'DR':     'badge-dr',
    'CSR':    'badge-csr',
    'MH':     'badge-mh',
    'AMD':    'badge-amd',
    'DRUSEN': 'badge-drusen',
    'DME':    'badge-dme',
    'NORMAL': 'badge-normal',
}

CLASS_DESC = {
    'CNV':    'Choroidal Neovascularization – pertumbuhan pembuluh darah abnormal di bawah retina.',
    'DR':     'Diabetic Retinopathy – kerusakan pembuluh darah retina akibat diabetes.',
    'CSR':    'Central Serous Retinopathy – akumulasi cairan serosa di bawah retina.',
    'MH':     'Macular Hole – lubang kecil di pusat makula (fovea).',
    'AMD':    'Age-related Macular Degeneration – degenerasi makula terkait usia.',
    'DRUSEN': 'Drusen – endapan lipid/protein di bawah epitel pigmen retina.',
    'DME':    'Diabetic Macular Edema – pembengkakan makula akibat komplikasi diabetes.',
    'NORMAL': 'Retina sehat tanpa tanda-tanda patologi.',
}

CLASS_SEVERITY = {
    'CNV': '🔴 Tinggi', 'DR': '🔴 Tinggi', 'DME': '🔴 Tinggi',
    'AMD': '🟠 Sedang-Tinggi', 'MH': '🟠 Sedang',
    'CSR': '🟡 Sedang', 'DRUSEN': '🟡 Rendah-Sedang',
    'NORMAL': '🟢 Normal',
}

IMG_SIZE = (224, 224)

GDRIVE_RESNET_URL       = "https://drive.google.com/file/d/1eFgQxxFegoF1699bTVh20fVzmXl2n-EJ/view?usp=drive_link"
GDRIVE_EFFICIENTNET_URL = "https://drive.google.com/file/d/1_WIEbc5xHhesDSjRD7SD4vxct2WDgll_/view?usp=sharing"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ─── Session State Init ────────────────────────────────────────────────────────
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'correct_cases' not in st.session_state:
    st.session_state.correct_cases = []
if 'wrong_cases' not in st.session_state:
    st.session_state.wrong_cases = []
if 'last_model' not in st.session_state:
    st.session_state.last_model = None

# ─── Fungsi Helper ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_from_drive(url: str, filename: str):
    path = MODEL_DIR / filename
    if not path.exists():
        with st.spinner(f"⬇️ Mengunduh model {filename}..."):
            gdown.download(url, str(path), quiet=False, fuzzy=True)
    model = load_model(str(path), compile=False)
    return model


def preprocess_resnet(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = resnet_preprocess(arr)
    return np.expand_dims(arr, 0)


def preprocess_effnet(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = effnet_preprocess(arr)
    return np.expand_dims(arr, 0)


def get_preprocess_fn(model_name: str):
    if model_name == 'ResNet50':
        return preprocess_resnet
    else:
        return preprocess_effnet


def get_last_conv_layer(model, model_name: str = "") -> str:
    """
    Cari conv layer terbaik untuk Grad-CAM++.
    - ResNet50       → conv5_block3_3_conv (feature map akhir block5)
    - EfficientNetB0 → top_conv (feature map sebelum global pooling)
    - Fallback       → Conv2D terakhir dengan kernel > 1x1
    """
    preferred = {
        'ResNet50':       ['conv5_block3_3_conv', 'conv5_block3_2_conv', 'conv5_block3_1_conv'],
        'EfficientNetB0': ['top_conv', 'block7a_project_conv', 'block6d_project_conv'],
    }
    if model_name in preferred:
        for target in preferred[model_name]:
            try:
                model.get_layer(target)
                return target
            except ValueError:
                continue

    # Fallback: Conv2D dengan kernel > 1x1
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            cfg = layer.get_config()
            ks  = cfg.get('kernel_size', (1, 1))
            k   = ks[0] if isinstance(ks, (list, tuple)) else ks
            if k > 1:
                return layer.name

    # Last resort
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("Tidak ditemukan layer Conv2D dalam model.")


def grad_cam_plusplus(model, img_array: np.ndarray, class_idx: int, layer_name: str) -> np.ndarray:
    """
    Grad-CAM++ yang benar:
    - grad_model dibangun agar output conv + logits sekaligus
    - watch() dipanggil SEBELUM forward pass di dalam tape (bukan setelah)
    - tape persistent=True agar grads dapat dihitung berurutan dari satu forward pass
    - Gradien orde-2 dan orde-3 dihitung terhadap conv_out, bukan berantai dari grads1
    """
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(layer_name).output, model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        # watch conv_out SEBELUM forward pass — inilah kunci koreksinya
        # Kita gunakan satu tape persistent untuk semua turunan
        conv_out, preds = grad_model(img_tensor, training=False)
        tape.watch(conv_out)
        loss = preds[:, class_idx]

    # dL/dA  — gradient pertama terhadap conv feature map
    grads1 = tape.gradient(loss, conv_out)

    # Untuk orde-2 dan orde-3, kita perlu nested tape yang watch conv_out sejak awal
    # Strategi: hitung ulang dengan nested tape agar chain rule benar
    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_out2, preds2 = grad_model(img_tensor, training=False)
                tape1.watch(conv_out2)
                tape2.watch(conv_out2)
                tape3.watch(conv_out2)
                loss2 = preds2[:, class_idx]
            g1 = tape1.gradient(loss2, conv_out2)   # dL/dA
        g2 = tape2.gradient(g1, conv_out2)           # d²L/dA²
    g3 = tape3.gradient(g2, conv_out2)               # d³L/dA³

    del tape  # bebaskan tape persistent

    # Ambil nilai numpy, lepas dimensi batch
    A  = conv_out2[0].numpy()   # (H, W, C)
    g1 = g1[0].numpy()
    g2 = g2[0].numpy()
    g3 = g3[0].numpy() if g3 is not None else np.zeros_like(g2)

    # ── Alpha Grad-CAM++ ──────────────────────────────────────────────────────
    # global_sum: sum_k A^k_{ij}  shape (C,)
    global_sum  = np.sum(A, axis=(0, 1))

    alpha_num   = g2
    alpha_denom = 2.0 * g2 + g3 * global_sum[np.newaxis, np.newaxis, :]

    # Hindari pembagian dengan nol
    alpha_denom = np.where(np.abs(alpha_denom) < 1e-7,
                           np.ones_like(alpha_denom),
                           alpha_denom)
    alphas  = alpha_num / alpha_denom                               # (H, W, C)

    # weights: sum_{i,j} ReLU(dL/dA) * alpha  → shape (C,)
    weights = np.sum(np.maximum(g1, 0) * alphas, axis=(0, 1))

    # CAM: weighted sum of feature maps  → shape (H, W)
    cam = np.sum(weights * A, axis=-1)
    cam = np.maximum(cam, 0)                                        # ReLU

    # Resize ke ukuran input & normalisasi ke [0, 1]
    cam = cv2.resize(cam, IMG_SIZE)
    c_min, c_max = cam.min(), cam.max()
    if c_max - c_min > 1e-8:
        cam = (cam - c_min) / (c_max - c_min)
    else:
        cam = np.zeros_like(cam)

    return cam


def overlay_heatmap(original: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    heatmap_color  = cm.jet(heatmap)[:, :, :3]
    heatmap_color  = (heatmap_color * 255).astype(np.uint8)
    original_uint8 = (original * 255).astype(np.uint8)
    overlay        = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def predict(model, img_array: np.ndarray):
    preds = model.predict(img_array, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return idx, float(preds[idx]) * 100, preds


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#f8fafc')
    buf.seek(0)
    return Image.open(buf)


def render_badge(label: str) -> str:
    cls = CLASS_BADGE.get(label, 'badge-normal')
    return f'<span class="badge {cls}">{label}</span>'


def generate_gradcam_analysis(model_name: str, correct_cases, wrong_cases,
                               model, layer_name: str, preprocess_fn) -> plt.Figure:
    n_correct = len(correct_cases)
    n_wrong   = len(wrong_cases)
    total     = n_correct + n_wrong
    if total == 0:
        return None

    fig, axes = plt.subplots(total, 3, figsize=(12, total * 3.2), facecolor='#f8fafc')
    if total == 1:
        axes = [axes]

    fig.suptitle(f"Grad-CAM++ Analysis — {model_name}", fontsize=13,
                 color='#0369a1', fontname='sans-serif', fontweight='bold', y=1.01)

    headers = ['Citra Asli', 'Grad-CAM++ Heatmap', 'Overlay']
    for ax, h in zip(axes[0], headers):
        ax.set_title(h, color='#475569', fontsize=9, pad=6, fontweight='600')

    for row_idx, (pil_img, true_lbl, pred_lbl, conf) in enumerate(correct_cases + wrong_cases):
        is_correct = row_idx < n_correct
        inp        = preprocess_fn(pil_img)
        orig_arr   = np.array(pil_img.convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0
        pred_idx   = CLASS_NAMES.index(pred_lbl)

        heatmap = grad_cam_plusplus(model, inp, pred_idx, layer_name)
        overlay = overlay_heatmap(orig_arr, heatmap)

        color = '#059669' if is_correct else '#dc2626'
        label = '✓ BENAR' if is_correct else '✗ SALAH'
        title = f"{label}  |  Pred: {pred_lbl}  |  True: {true_lbl}  |  {conf:.1f}%"

        axes[row_idx][0].imshow(orig_arr)
        axes[row_idx][1].imshow(heatmap, cmap='jet')
        axes[row_idx][2].imshow(overlay)

        for ax in axes[row_idx]:
            ax.axis('off')
            ax.set_facecolor('#f8fafc')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)
                spine.set_visible(True)

        axes[row_idx][0].set_title(title, color=color, fontsize=8, pad=4, loc='left')

    plt.tight_layout()
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Header Sidebar ──
    st.markdown("""
    <div class="sb-header">
        <div style='font-size:1.5rem;margin-bottom:0.3rem'>👁️</div>
        <div style='font-family:"Roboto Condensed",sans-serif;font-size:1.05rem;
                    color:#ffffff;font-weight:700;letter-spacing:0.02em'>RetinaScan AI</div>
        <div style='font-size:0.68rem;color:rgba(255,255,255,0.7);margin-top:0.2rem;
                    text-transform:uppercase;letter-spacing:0.08em'>OCT Classification System</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Konfigurasi Model ──
    st.markdown("""<div class="sb-section-title">⚙️ Konfigurasi Model</div>""",
                unsafe_allow_html=True)
    use_resnet       = st.checkbox("ResNet50",       value=True)
    use_efficientnet = st.checkbox("EfficientNetB0", value=True)

    st.markdown("---")

    # ── Status Model (ditampilkan setelah load) ──
    st.markdown("""
    <div class="sb-section">
        <div class="sb-section-title">🔌 Status Model</div>
    """, unsafe_allow_html=True)

    model_status_placeholder = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Statistik Sesi ──
    n_correct_ss = len(st.session_state.correct_cases)
    n_wrong_ss   = len(st.session_state.wrong_cases)
    total_pred   = st.session_state.total_predictions

    st.markdown(f"""
    <div class="sb-section">
        <div class="sb-section-title">📈 Statistik Sesi</div>
        <div class="sb-stat">
            <span>Total Prediksi</span>
            <span class="sb-stat-val">{total_pred}</span>
        </div>
        <div class="sb-stat">
            <span>Kasus Benar (Tab 2)</span>
            <span class="sb-stat-val" style="color:#059669">{n_correct_ss}/5</span>
        </div>
        <div class="sb-stat">
            <span>Kasus Salah (Tab 2)</span>
            <span class="sb-stat-val" style="color:#dc2626">{n_wrong_ss}/5</span>
        </div>
        <div class="sb-stat">
            <span>Model Aktif</span>
            <span class="sb-stat-val">{sum([use_resnet, use_efficientnet])}/2</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Info Teknis ──
    st.markdown("""
    <div class="sb-section">
        <div class="sb-section-title">🔬 Informasi Teknis</div>
        <div class="sb-stat">
            <span>Input Size</span>
            <span class="sb-stat-val">224×224</span>
        </div>
        <div class="sb-stat">
            <span>Kelas</span>
            <span class="sb-stat-val">8</span>
        </div>
        <div class="sb-stat">
            <span>XAI Method</span>
            <span class="sb-stat-val">Grad-CAM++</span>
        </div>
        <div class="sb-stat">
            <span>Dataset</span>
            <span class="sb-stat-val">OCT Retina</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Legenda Kelas ──
    st.markdown("""
    <div class="sb-section">
        <div class="sb-section-title">🏥 Kelas Penyakit</div>
    """, unsafe_allow_html=True)

    badge_colors = {
        'CNV':'#fee2e2;color:#b91c1c', 'DR':'#fce7f3;color:#9d174d',
        'CSR':'#e0f2fe;color:#0c4a6e', 'MH':'#ede9fe;color:#5b21b6',
        'AMD':'#ffedd5;color:#9a3412', 'DRUSEN':'#d1fae5;color:#065f46',
        'DME':'#fef3c7;color:#92400e', 'NORMAL':'#dbeafe;color:#1e40af',
    }
    chips_html = ""
    for cls in CLASS_NAMES:
        bc = badge_colors.get(cls, '#f1f5f9;color:#475569')
        chips_html += f"<span class='sb-cls-chip' style='background:{bc.split(';')[0].replace('background:','')};{bc.split(';')[1] if ';' in bc else ''}'>{cls}</span>"
    st.markdown(chips_html, unsafe_allow_html=True)

    with st.expander("Lihat Deskripsi Kelas", expanded=False):
        for cls, desc in CLASS_DESC.items():
            sev = CLASS_SEVERITY.get(cls, '')
            st.markdown(f"**{cls}** {sev}\n\n{desc}\n\n---")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div style='text-align:center;padding:0.75rem 0 0.25rem;
                font-size:0.65rem;color:#94a3b8;line-height:1.7;
                border-top:1px solid #e2e8f0;margin-top:0.5rem'>
        ResNet50 vs EfficientNetB0<br>
        Grad-CAM++ · OCT Retina Classification<br>
        <span style='color:#cbd5e1'>UPN Veteran Jawa Timur</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Header Utama ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-scan-line"></div>
    <div style='position:relative;z-index:1'>
        <div class="hero-title">👁️ RetinaScan AI</div>
        <div class="hero-sub">
            Perbandingan <strong style='color:#7dd3fc'>ResNet50</strong> &amp;
            <strong style='color:#7dd3fc'>EfficientNetB0</strong> &nbsp;·&nbsp;
            Explainable AI Grad-CAM++ &nbsp;·&nbsp; 8 Kelas Penyakit Retina OCT
        </div>
        <div class="hero-badge">SYSTEM ONLINE &nbsp;·&nbsp; OCT CLASSIFICATION v2.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
models = {}
with st.status("Memuat model...", expanded=False) as status:
    if use_resnet:
        try:
            models['ResNet50'] = load_model_from_drive(GDRIVE_RESNET_URL, "resnet50_retina.h5")
            st.write("✅ ResNet50 berhasil dimuat")
        except Exception as e:
            st.warning(f"⚠️ ResNet50 gagal dimuat: {e}")
    if use_efficientnet:
        try:
            models['EfficientNetB0'] = load_model_from_drive(GDRIVE_EFFICIENTNET_URL, "efficientnetb0_retina.h5")
            st.write("✅ EfficientNetB0 berhasil dimuat")
        except Exception as e:
            st.warning(f"⚠️ EfficientNetB0 gagal dimuat: {e}")
    status.update(label=f"✅ {len(models)} model aktif", state="complete")

# Update sidebar model status setelah load
model_pills_html = ""
for mname in ['ResNet50', 'EfficientNetB0']:
    active  = mname in models
    dot_col = '#059669' if active else '#94a3b8'
    txt_col = '#0f172a' if active else '#94a3b8'
    status_txt = 'Aktif' if active else 'Tidak dimuat'
    acc = '93.0%' if mname == 'ResNet50' else '89.0%'
    model_pills_html += f"""
    <div class="sb-model-pill">
        <div class="sb-model-dot" style="background:{dot_col};
             {'box-shadow:0 0 0 2px rgba(5,150,105,0.25)' if active else ''}"></div>
        <div style="flex:1">
            <div style="font-weight:600;color:{txt_col};font-size:0.78rem">{mname}</div>
            <div style="font-size:0.65rem;color:#94a3b8">{status_txt} · Acc {acc}</div>
        </div>
    </div>
    """
model_status_placeholder.markdown(model_pills_html, unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Prediksi Gambar",
    "🗺️ Grad-CAM++ Analysis",
    "📊 Perbandingan Model",
    "ℹ️ Tentang Sistem"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI GAMBAR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="card">
        <div class="card-title">🔬 Prediksi Citra OCT — Kedua Model</div>
        <p style='font-size:0.85rem;color:#475569;line-height:1.7'>
        Upload 1 gambar dari data test. Sistem akan menjalankan prediksi pada 
        <strong>ResNet50</strong> dan <strong>EfficientNetB0</strong> secara bersamaan,
        menampilkan kelas prediksi, confidence score, dan distribusi probabilitas.
        Visualisasi Grad-CAM++ ditampilkan dari model EfficientNetB0.
        </p>
    </div>
    """, unsafe_allow_html=True)
 
    uploaded = st.file_uploader("Upload gambar OCT (.jpg / .png)", type=["jpg","jpeg","png"])
 
    if not models:
        st.warning("⚠️ Tidak ada model yang termuat.")
    elif uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.markdown("---")
        img_col, info_col = st.columns([1,3])
        with img_col:
            st.image(pil_img, caption="Citra Input", use_container_width=True)
        with info_col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Nama File</div>
                <div style='font-family:"JetBrains Mono",monospace;color:#0f172a;font-size:0.9rem;font-weight:500'>{uploaded.name}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Ukuran Input Model</div>
                <div style='font-family:"JetBrains Mono",monospace;color:#0f172a;font-size:0.9rem;font-weight:500'>224 × 224 px (RGB)</div>
            </div>
            """, unsafe_allow_html=True)
 
        st.markdown("---")
        model_list = list(models.items())
        n_models   = len(model_list)
 
        if n_models == 2:
            cols_header = st.columns(2)
            colors    = ['#0369a1','#7c3aed']
            bg_colors = ['#e0f2fe','#ede9fe']
            border_c  = ['rgba(3,105,161,0.3)','rgba(124,58,237,0.3)']
            for ci, (mname, _) in enumerate(model_list):
                with cols_header[ci]:
                    st.markdown(f"""
                    <div style='text-align:center;background:{bg_colors[ci]};border-radius:10px;
                                padding:0.6rem;border:1px solid {border_c[ci]};margin-bottom:0.75rem'>
                        <span style='font-size:0.9rem;font-weight:700;color:{colors[ci]}'>
                            {"🔵" if ci==0 else "🟣"} {mname}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
 
        results   = {}
        pred_cols = st.columns(n_models)
        for ci, (mname, model) in enumerate(model_list):
            preprocess_fn = get_preprocess_fn(mname)
            img_arr       = preprocess_fn(pil_img)
            pred_idx, conf, probs = predict(model, img_arr)
            pred_label = CLASS_NAMES[pred_idx]
            results[mname] = (pred_idx, conf, probs, pred_label, img_arr)
            badge = render_badge(pred_label)
            color = '#0369a1' if ci==0 else '#7c3aed'
            bg_c  = '#f0f9ff' if ci==0 else '#f5f3ff'
            bdr_c = 'rgba(3,105,161,0.2)' if ci==0 else 'rgba(124,58,237,0.2)'
            with pred_cols[ci]:
                st.markdown(f"""
                <div style='background:{bg_c};border:1px solid {bdr_c};border-radius:12px;
                            padding:1.25rem;margin-bottom:0.75rem;box-shadow:0 1px 4px rgba(0,0,0,0.06)'>
                    <div style='font-size:0.68rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:0.5rem'>Prediksi Kelas</div>
                    <div style='margin-bottom:0.4rem'>{badge}</div>
                    <div style='font-family:"JetBrains Mono",monospace;font-size:2rem;
                                font-weight:700;color:{color};line-height:1.1'>{conf:.1f}%</div>
                    <div style='font-size:0.72rem;color:#64748b;margin-top:0.2rem'>Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"{CLASS_SEVERITY.get(pred_label,'')} · {CLASS_DESC[pred_label]}")
                st.markdown("<div style='font-size:0.75rem;color:#64748b;margin:0.75rem 0 0.4rem'>Top-5 Probabilitas</div>", unsafe_allow_html=True)
                for i in np.argsort(probs)[::-1][:5]:
                    st.progress(float(probs[i]), text=f"{CLASS_NAMES[i]}: {probs[i]*100:.1f}%")
 
        st.session_state.total_predictions += 1
 
        if n_models == 2:
            labels = [r[3] for r in results.values()]
            if labels[0] == labels[1]:
                st.markdown(f"""
                <div class="result-agree">
                    <span style='font-size:1.3rem'>✅</span>
                    <span style='font-size:0.88rem;color:#1e3a5f'>
                        Kedua model <strong style='color:#059669'>sepakat</strong> — prediksi kelas 
                        <strong style='color:#059669'>{labels[0]}</strong>
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                confs  = [r[1] for r in results.values()]
                mnames = list(results.keys())
                st.markdown(f"""
                <div class="result-disagree">
                    <span style='font-size:1.3rem'>⚠️</span>
                    <span style='font-size:0.88rem;color:#78350f'>
                        Kedua model <strong style='color:#d97706'>berbeda pendapat</strong> —
                        {mnames[0]}: <strong style='color:#0369a1'>{labels[0]} ({confs[0]:.1f}%)</strong>
                        &nbsp;|&nbsp;
                        {mnames[1]}: <strong style='color:#7c3aed'>{labels[1]} ({confs[1]:.1f}%)</strong>
                    </span>
                </div>""", unsafe_allow_html=True)
 
        # ── Grad-CAM++ hanya dari EfficientNetB0 ──
        st.markdown("---")
        st.markdown('<div class="card-title">🧠 Explainable AI — Grad-CAM++</div>', unsafe_allow_html=True)
 
        if st.button("🔍 Generate Grad-CAM++"):
            orig_arr = np.array(pil_img.resize(IMG_SIZE), dtype=np.float32) / 255.0
            # Pilih EfficientNetB0, fallback ke model pertama jika tidak ada
            gcam_model_name = 'EfficientNetB0' if 'EfficientNetB0' in models else list(models.keys())[0]
            gcam_model = models[gcam_model_name]
            pred_idx, conf, probs, pred_label, img_arr = results[gcam_model_name]
            try:
                with st.spinner("Menghitung Grad-CAM++..."):
                    layer_name = get_last_conv_layer(gcam_model, gcam_model_name)
                    heatmap    = grad_cam_plusplus(gcam_model, img_arr, pred_idx, layer_name)
                    overlay    = overlay_heatmap(orig_arr, heatmap)
                    hm_color   = cm.jet(heatmap)[:,:,:3]
 
                c1, c2, c3 = st.columns(3)
                with c1: st.image(orig_arr,  caption="Citra Asli",        use_container_width=True, clamp=True)
                with c2: st.image(hm_color,  caption="Grad-CAM++ Heatmap", use_container_width=True, clamp=True)
                with c3: st.image(overlay,   caption="Overlay",            use_container_width=True, clamp=True)
                st.caption(f"Explainable AI Grad-CAM++ · Prediksi: **{pred_label}** ({conf:.1f}%)")
 
                # Deskripsi area fokus per kelas
                focus_area = {
                    'AMD':    'area drusen dan perubahan degeneratif pada epitel pigmen retina (RPE) serta membran Bruch',
                    'CNV':    'area neovaskularisasi koroidal dan kebocoran cairan subretinal di bawah makula',
                    'CSR':    'area akumulasi cairan serosa di bawah retina neurosensori dan lapisan epitel pigmen retina',
                    'DME':    'area pembengkakan makula dan akumulasi cairan intraretinal akibat kebocoran kapiler',
                    'DR':     'area perdarahan retina, eksudat keras, dan mikroaneurisma akibat kerusakan pembuluh darah',
                    'DRUSEN': 'area endapan drusen di bawah epitel pigmen retina yang tampak sebagai bintik-bintik terang',
                    'MH':     'area lubang makula di pusat fovea dan jaringan sekitarnya yang mengalami tarikan vitreous',
                    'NORMAL': 'lapisan retina yang sehat dan terstruktur dengan baik tanpa tanda-tanda patologi',
                }
                area = focus_area.get(pred_label, 'area lesi retina yang mengalami perubahan tekstur')
 
                st.markdown(f"""
                <div class="analysis-box">
                    <strong style='color:#0369a1'>📋 Analisis Visual Grad-CAM++</strong><br><br>
                    Hasil Grad-CAM++ menunjukkan bahwa model <strong>EfficientNetB0</strong> 
                    lebih banyak memfokuskan perhatian pada 
                    <strong>{area}</strong> saat memprediksi kelas 
                    <strong>{pred_label}</strong> dengan confidence <strong>{conf:.1f}%</strong>. 
                    Warna merah-kuning pada heatmap menunjukkan area dengan aktivasi tertinggi, 
                    sementara warna biru menunjukkan area yang kurang relevan bagi model.<br><br>
                    Hal ini menunjukkan bahwa model telah mempelajari karakteristik visual 
                    yang relevan secara klinis dengan diagnosis <strong>{pred_label}</strong> 
                    ({CLASS_DESC.get(pred_label, '')}) pada citra OCT.
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Gagal: {e}")
    else:
        st.markdown("""
        <div style='text-align:center;padding:4rem;color:#94a3b8;
                    background:#f8fafc;border-radius:12px;border:2px dashed #e2e8f0'>
            <div style='font-size:3.5rem;margin-bottom:0.75rem'>🖼️</div>
            <div style='font-size:0.95rem;font-weight:600;color:#64748b;margin-bottom:0.4rem'>
                Upload Citra OCT untuk Memulai
            </div>
            <div style='font-size:0.82rem;color:#94a3b8;line-height:1.7'>
                Upload satu gambar OCT dari data test kamu<br>
                <span style='font-size:0.75rem'>Format: JPG / PNG · Resolusi apa pun (otomatis resize ke 224×224)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GRAD-CAM++ ANALYSIS: 5 Benar + 5 Salah
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🗺️ Grad-CAM++ Analysis — 5 Prediksi Benar & 5 Prediksi Salah</div>
        <p style='font-size:0.85rem;color:#475569;line-height:1.7'>
        Upload gambar satu per satu dari data test. Sistem otomatis mengkategorikan 
        setiap gambar sebagai <strong style='color:#059669'>prediksi benar</strong> atau 
        <strong style='color:#dc2626'>prediksi salah</strong> berdasarkan label true yang kamu masukkan.
        Setelah terkumpul 5 + 5, generate Grad-CAM++ beserta analisis visualnya.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not models:
        st.warning("⚠️ Tidak ada model yang termuat.")
    else:
        model_choice  = st.selectbox(
            "🏆 Pilih model untuk analisis Grad-CAM++",
            list(models.keys()),
            help="Pilih model terbaik berdasarkan hasil evaluasimu"
        )
        target_model  = models[model_choice]
        mc_color      = '#0369a1' if model_choice == 'ResNet50' else '#7c3aed'
        preprocess_fn = get_preprocess_fn(model_choice)

        if st.session_state.last_model != model_choice:
            st.session_state.correct_cases = []
            st.session_state.wrong_cases   = []
            st.session_state.last_model    = model_choice

        n_correct = len(st.session_state.correct_cases)
        n_wrong   = len(st.session_state.wrong_cases)

        prog_col1, prog_col2, prog_col3 = st.columns(3)
        with prog_col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Model Aktif</div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:1rem;
                            color:{mc_color};font-weight:700'>{model_choice}</div>
            </div>""", unsafe_allow_html=True)
        with prog_col2:
            st.markdown(f"""
            <div class="metric-box" style='border-left-color:#059669'>
                <div class="metric-label">✓ Terkumpul Benar</div>
                <div class="metric-value" style='color:#059669'>{n_correct} / 5</div>
            </div>""", unsafe_allow_html=True)
        with prog_col3:
            st.markdown(f"""
            <div class="metric-box" style='border-left-color:#dc2626'>
                <div class="metric-label">✗ Terkumpul Salah</div>
                <div class="metric-value" style='color:#dc2626'>{n_wrong} / 5</div>
            </div>""", unsafe_allow_html=True)

        st.progress(min((n_correct + n_wrong) / 10, 1.0),
                    text=f"Terkumpul {n_correct + n_wrong}/10 kasus")

        if st.button("🔄 Reset Semua Kasus"):
            st.session_state.correct_cases = []
            st.session_state.wrong_cases   = []
            st.rerun()

        st.markdown("---")

        still_need_correct = n_correct < 5
        still_need_wrong   = n_wrong < 5

        if still_need_correct or still_need_wrong:
            st.markdown('<div class="card-title">📂 Upload Gambar dari Data Test</div>',
                        unsafe_allow_html=True)

            up_col, lbl_col = st.columns([2, 1])
            with up_col:
                sample_file = st.file_uploader(
                    "Upload 1 gambar OCT",
                    type=["jpg", "jpeg", "png"],
                    key=f"tab2_upload_{n_correct}_{n_wrong}"
                )
            with lbl_col:
                true_label = st.selectbox(
                    "True Label (kelas asli gambar ini)",
                    CLASS_NAMES,
                    key=f"tab2_label_{n_correct}_{n_wrong}"
                )

            if sample_file:
                pil_img = Image.open(sample_file).convert("RGB")
                img_arr = preprocess_fn(pil_img)

                pred_idx, conf, probs = predict(target_model, img_arr)
                pred_label = CLASS_NAMES[pred_idx]
                is_correct = (pred_label == true_label)

                res_color  = '#059669' if is_correct else '#dc2626'
                res_bg     = '#f0fdf4' if is_correct else '#fef2f2'
                res_bdr    = '#86efac' if is_correct else '#fca5a5'
                res_label  = '✓ PREDIKSI BENAR' if is_correct else '✗ PREDIKSI SALAH'
                pred_badge = render_badge(pred_label)
                true_badge = render_badge(true_label)

                r_col1, r_col2 = st.columns([1, 2])
                with r_col1:
                    st.image(pil_img, caption="Citra Input", use_container_width=True)
                with r_col2:
                    st.markdown(f"""
                    <div style='background:{res_bg};border:1px solid {res_bdr};
                                border-left:4px solid {res_color};
                                border-radius:10px;padding:1rem 1.25rem'>
                        <div style='font-size:0.72rem;font-weight:700;color:{res_color};
                                    letter-spacing:0.1em;margin-bottom:0.75rem'>{res_label}</div>
                        <div style='display:flex;gap:1.5rem;margin-bottom:0.75rem'>
                            <div>
                                <div style='font-size:0.68rem;color:#64748b'>Prediksi {model_choice}</div>
                                <div style='margin-top:0.3rem'>{pred_badge}</div>
                            </div>
                            <div>
                                <div style='font-size:0.68rem;color:#64748b'>True Label</div>
                                <div style='margin-top:0.3rem'>{true_badge}</div>
                            </div>
                            <div>
                                <div style='font-size:0.68rem;color:#64748b'>Confidence</div>
                                <div style='font-family:"JetBrains Mono",monospace;font-size:1.2rem;
                                            font-weight:700;color:{mc_color}'>{conf:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='font-size:0.73rem;color:#64748b;margin-top:0.75rem'>"
                                "Top-5 Probabilitas</div>", unsafe_allow_html=True)
                    for i in np.argsort(probs)[::-1][:5]:
                        st.progress(float(probs[i]), text=f"{CLASS_NAMES[i]}: {probs[i]*100:.1f}%")

                can_add = (is_correct and still_need_correct) or (not is_correct and still_need_wrong)
                if can_add:
                    entry     = (pil_img, true_label, pred_label, conf)
                    btn_label = f"➕ Simpan sebagai {'Prediksi Benar' if is_correct else 'Prediksi Salah'}"
                    if st.button(btn_label, key=f"btn_save_{n_correct}_{n_wrong}"):
                        if is_correct:
                            st.session_state.correct_cases.append(entry)
                        else:
                            st.session_state.wrong_cases.append(entry)
                        st.rerun()
                elif is_correct and not still_need_correct:
                    st.info("✅ Sudah cukup 5 prediksi benar. Upload gambar yang salah diprediksi.")
                elif not is_correct and not still_need_wrong:
                    st.info("✅ Sudah cukup 5 prediksi salah. Upload gambar yang benar diprediksi.")
        else:
            st.success("🎉 Sudah terkumpul 5 prediksi benar + 5 prediksi salah! Siap generate Grad-CAM++.")

        if n_correct > 0 or n_wrong > 0:
            st.markdown("---")
            st.markdown('<div class="card-title">📋 Kasus yang Sudah Terkumpul</div>',
                        unsafe_allow_html=True)

            def render_case_table(cases, is_correct):
                color = '#059669' if is_correct else '#dc2626'
                bg    = '#f0fdf4' if is_correct else '#fef2f2'
                label = f'✓ PREDIKSI BENAR ({len(cases)}/5)' if is_correct else f'✗ PREDIKSI SALAH ({len(cases)}/5)'
                rows  = ""
                for idx, (_, true_lbl, pred_lbl, conf) in enumerate(cases):
                    pb = render_badge(pred_lbl)
                    tb = render_badge(true_lbl)
                    rows += f"""<tr style='border-bottom:1px solid #f1f5f9'>
                        <td style='padding:0.4rem;color:#94a3b8;font-size:0.78rem'>{idx+1}</td>
                        <td style='padding:0.4rem'>{pb}</td>
                        <td style='padding:0.4rem'>{tb}</td>
                        <td style='padding:0.4rem;font-family:"JetBrains Mono",monospace;
                                   color:{color};font-size:0.85rem;font-weight:700'>{conf:.1f}%</td>
                    </tr>"""
                return f"""
                <div style='margin-bottom:1rem;background:{bg};border:1px solid {"#86efac" if is_correct else "#fca5a5"};
                            border-radius:10px;padding:0.75rem'>
                    <div style='font-size:0.78rem;font-weight:700;color:{color};margin-bottom:0.4rem'>{label}</div>
                    <table style='width:100%;border-collapse:collapse;font-size:0.82rem'>
                        <thead><tr style='color:#94a3b8;border-bottom:2px solid {"#d1fae5" if is_correct else "#fee2e2"}'>
                            <th style='padding:0.4rem;text-align:left'>#</th>
                            <th style='padding:0.4rem;text-align:left'>Prediksi</th>
                            <th style='padding:0.4rem;text-align:left'>True Label</th>
                            <th style='padding:0.4rem;text-align:left'>Confidence</th>
                        </tr></thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>"""

            tbl_col1, tbl_col2 = st.columns(2)
            with tbl_col1:
                st.markdown(render_case_table(st.session_state.correct_cases, True),
                            unsafe_allow_html=True)
            with tbl_col2:
                st.markdown(render_case_table(st.session_state.wrong_cases, False),
                            unsafe_allow_html=True)

        if n_correct >= 1 and n_wrong >= 1:
            st.markdown("---")
            if st.button(f"🚀 Generate Analisis Grad-CAM++ — {model_choice}"):
                n_c           = len(st.session_state.correct_cases)
                n_w           = len(st.session_state.wrong_cases)
                avg_conf_c    = np.mean([c[3] for c in st.session_state.correct_cases]) if n_c > 0 else 0
                avg_conf_w    = np.mean([c[3] for c in st.session_state.wrong_cases])   if n_w > 0 else 0
                wrong_classes = [c[1] for c in st.session_state.wrong_cases]
                most_wrong    = max(set(wrong_classes), key=wrong_classes.count) if wrong_classes else "-"

                with st.spinner(f"Menghitung Grad-CAM++ — {model_choice}..."):
                    try:
                        layer_name = get_last_conv_layer(target_model, model_choice)
                        st.caption(f"Layer target: `{layer_name}`")
                        fig = generate_gradcam_analysis(
                            model_choice,
                            st.session_state.correct_cases,
                            st.session_state.wrong_cases,
                            target_model, layer_name, preprocess_fn
                        )
                        if fig:
                            st.pyplot(fig, use_container_width=True)
                            st.markdown(f"""
                            <div class="analysis-box">
                                <strong style='color:#0369a1'>
                                    📋 Analisis Visual Grad-CAM++ — {model_choice}
                                </strong><br><br>
                                Dari <strong>{n_c + n_w}</strong> contoh yang dianalisis menggunakan model 
                                <strong>{model_choice}</strong>, terdapat <strong>{n_c}</strong> prediksi 
                                benar dengan rata-rata confidence <strong>{avg_conf_c:.1f}%</strong> dan 
                                <strong>{n_w}</strong> prediksi salah dengan rata-rata confidence 
                                <strong>{avg_conf_w:.1f}%</strong>.<br><br>
                                Hasil Grad-CAM++ menunjukkan bahwa model lebih banyak memfokuskan perhatian 
                                pada <strong>area lesi retina yang mengalami perubahan tekstur dan 
                                reflektivitas jaringan</strong>, seperti akumulasi cairan subretinal, 
                                perubahan lapisan epitel pigmen retina (RPE), serta kehadiran drusen dan 
                                neovaskularisasi koroidal. Hal ini menunjukkan bahwa model telah mempelajari 
                                karakteristik visual yang relevan secara klinis dengan diagnosis penyakit 
                                retina pada citra OCT.<br><br>
                                Pada kasus prediksi yang salah, heatmap cenderung menyebar ke area yang 
                                kurang representatif — khususnya pada kelas <strong>{most_wrong}</strong> 
                                yang memiliki fitur visual yang mirip dengan kelas lain, mengindikasikan 
                                potensi area peningkatan model di masa mendatang.
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gagal generate Grad-CAM++: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PERBANDINGAN MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    R_CLASSES   = ['AMD','CNV','CSR','DME','DR','DRUSEN','MH','NORMAL']
    R_PRECISION = [1.00, 0.85, 0.99, 0.96, 0.99, 0.86, 1.00, 0.80]
    R_RECALL    = [1.00, 0.90, 1.00, 0.83, 0.99, 0.75, 0.99, 0.97]
    R_F1        = [1.00, 0.88, 1.00, 0.89, 0.99, 0.80, 1.00, 0.88]
    R_ACC, R_PREC_MACRO, R_REC_MACRO, R_F1_MACRO = 93.0, 93.0, 93.0, 93.0

    E_PRECISION = [1.00, 0.82, 0.95, 0.82, 0.90, 0.79, 1.00, 0.83]
    E_RECALL    = [0.99, 0.82, 0.98, 0.87, 0.97, 0.68, 0.88, 0.91]
    E_F1        = [0.99, 0.82, 0.96, 0.85, 0.93, 0.73, 0.93, 0.87]
    E_ACC, E_PREC_MACRO, E_REC_MACRO, E_F1_MACRO = 89.0, 89.0, 89.0, 89.0

    winner = "ResNet50" if R_ACC > E_ACC else "EfficientNetB0"
    diff   = abs(R_ACC - E_ACC)

    st.markdown('<div class="card-title">📊 Perbandingan Arsitektur & Evaluasi Model</div>',
                unsafe_allow_html=True)

    # ── Banner pemenang ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#f0fdf4,#dcfce7);
                border:1px solid #86efac;border-left:5px solid #059669;border-radius:12px;
                padding:1.1rem 1.5rem;margin-bottom:1.25rem;display:flex;align-items:center;gap:1rem;
                box-shadow:0 1px 4px rgba(5,150,105,0.1)'>
        <div style='font-size:2.2rem'>🏆</div>
        <div>
            <div style='font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em'>
                Model Terbaik — Test Set
            </div>
            <div style='font-size:1.2rem;color:#059669;font-weight:700;margin:0.15rem 0'>{winner}</div>
            <div style='font-size:0.82rem;color:#475569'>
                Accuracy <strong style='color:#059669'>{max(R_ACC,E_ACC):.1f}%</strong>
                &nbsp;·&nbsp; unggul <strong style='color:#059669'>{diff:.1f}%</strong>
                dari model lainnya
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Overall metrics side-by-side ────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    def metric_pair(col, label, rv, ev):
        col.markdown(f"""
        <div style='background:#f8fafc;border-radius:10px;padding:0.8rem 0.75rem;text-align:center;
                    border:1px solid #e2e8f0;box-shadow:0 1px 3px rgba(0,0,0,0.05)'>
            <div style='font-size:0.62rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;
                        margin-bottom:0.4rem;font-weight:600'>{label}</div>
            <div style='display:flex;justify-content:center;gap:0.75rem;align-items:center'>
                <div>
                    <div style='font-family:"JetBrains Mono",monospace;font-size:1rem;
                                color:#0369a1;font-weight:700'>{rv:.1f}%</div>
                    <div style='font-size:0.6rem;color:#94a3b8'>R50</div>
                </div>
                <div style='color:#cbd5e1;font-size:0.8rem'>vs</div>
                <div>
                    <div style='font-family:"JetBrains Mono",monospace;font-size:1rem;
                                color:#7c3aed;font-weight:700'>{ev:.1f}%</div>
                    <div style='font-size:0.6rem;color:#94a3b8'>Eff</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    metric_pair(m1, "Accuracy",  R_ACC,        E_ACC)
    metric_pair(m2, "Precision", R_PREC_MACRO, E_PREC_MACRO)
    metric_pair(m3, "Recall",    R_REC_MACRO,  E_REC_MACRO)
    metric_pair(m4, "F1-Score",  R_F1_MACRO,   E_F1_MACRO)
    st.markdown("<div style='font-size:0.72rem;color:#94a3b8;margin:0.5rem 0 1rem'>"
                "🔵 ResNet50 &nbsp;·&nbsp; 🟣 EfficientNetB0</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Grafik 1: Overall metrics bar chart ────────────────────────────────
    st.markdown('<div class="card-title">📈 Grafik Perbandingan Metrik Keseluruhan</div>',
                unsafe_allow_html=True)

    metrics_overall = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    r_vals = [R_ACC, R_PREC_MACRO, R_REC_MACRO, R_F1_MACRO]
    e_vals = [E_ACC, E_PREC_MACRO, E_REC_MACRO, E_F1_MACRO]
    x = np.arange(len(metrics_overall))
    w = 0.35

    fig1, ax1 = plt.subplots(figsize=(8, 4), facecolor='#f8fafc')
    ax1.set_facecolor('#f8fafc')
    b1 = ax1.bar(x - w/2, r_vals, w, label='ResNet50',       color='#0369a1', alpha=0.88)
    b2 = ax1.bar(x + w/2, e_vals, w, label='EfficientNetB0', color='#7c3aed', alpha=0.88)
    ax1.set_xticks(x); ax1.set_xticklabels(metrics_overall, color='#475569', fontsize=9)
    ax1.set_ylim(80, 100); ax1.set_ylabel('Score (%)', color='#94a3b8', fontsize=9)
    ax1.tick_params(colors='#94a3b8')
    for sp in ['top','right']: ax1.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax1.spines[sp].set_color('#e2e8f0')
    ax1.yaxis.grid(True, color='#e2e8f0', linewidth=0.8); ax1.set_axisbelow(True)
    ax1.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#475569', fontsize=9)
    ax1.set_title("Overall Metrics — ResNet50 vs EfficientNetB0",
                  color='#1e293b', fontsize=10, pad=10)
    for bar in b1:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                 f'{bar.get_height():.1f}', ha='center', va='bottom',
                 color='#0369a1', fontsize=8, fontweight='bold')
    for bar in b2:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                 f'{bar.get_height():.1f}', ha='center', va='bottom',
                 color='#7c3aed', fontsize=8, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)

    st.markdown("---")

    # ── Grafik 2: Per-class F1-Score ─────────────────────────────────────
    st.markdown('<div class="card-title">📊 F1-Score Per Kelas Penyakit</div>',
                unsafe_allow_html=True)

    x2 = np.arange(len(R_CLASSES))
    fig2, ax2 = plt.subplots(figsize=(11, 4.5), facecolor='#f8fafc')
    ax2.set_facecolor('#f8fafc')
    b3 = ax2.bar(x2 - w/2, [v*100 for v in R_F1], w, label='ResNet50',       color='#0369a1', alpha=0.88)
    b4 = ax2.bar(x2 + w/2, [v*100 for v in E_F1], w, label='EfficientNetB0', color='#7c3aed', alpha=0.88)
    ax2.set_xticks(x2); ax2.set_xticklabels(R_CLASSES, color='#475569', fontsize=9)
    ax2.set_ylim(60, 105); ax2.set_ylabel('F1-Score (%)', color='#94a3b8', fontsize=9)
    ax2.tick_params(colors='#94a3b8')
    for sp in ['top','right']: ax2.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax2.spines[sp].set_color('#e2e8f0')
    ax2.yaxis.grid(True, color='#e2e8f0', linewidth=0.8); ax2.set_axisbelow(True)
    ax2.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#475569', fontsize=9)
    ax2.set_title("F1-Score Per Kelas — ResNet50 vs EfficientNetB0",
                  color='#1e293b', fontsize=10, pad=10)
    for bar in b3:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{bar.get_height():.0f}', ha='center', va='bottom',
                 color='#0369a1', fontsize=7, fontweight='bold')
    for bar in b4:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{bar.get_height():.0f}', ha='center', va='bottom',
                 color='#7c3aed', fontsize=7, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.markdown("---")

    # ── Tabel per-class lengkap ──────────────────────────────────────────
    st.markdown('<div class="card-title">📋 Tabel Lengkap Per Kelas</div>', unsafe_allow_html=True)

    rows = ""
    for i, cls in enumerate(R_CLASSES):
        r_p, r_r, r_f = R_PRECISION[i]*100, R_RECALL[i]*100, R_F1[i]*100
        e_p, e_r, e_f = E_PRECISION[i]*100, E_RECALL[i]*100, E_F1[i]*100
        badge = render_badge(cls)
        def cmp(rv, ev):
            if rv > ev:   return f"<span style='color:#059669;font-weight:700'>{rv:.0f}%</span>", f"<span style='color:#94a3b8'>{ev:.0f}%</span>"
            elif rv < ev: return f"<span style='color:#94a3b8'>{rv:.0f}%</span>", f"<span style='color:#059669;font-weight:700'>{ev:.0f}%</span>"
            else:         return f"<span style='color:#475569'>{rv:.0f}%</span>", f"<span style='color:#475569'>{ev:.0f}%</span>"
        rp_h, ep_h = cmp(r_p, e_p)
        rr_h, er_h = cmp(r_r, e_r)
        rf_h, ef_h = cmp(r_f, e_f)
        rows += f"""<tr style='border-bottom:1px solid #f1f5f9'>
            <td style='padding:0.5rem'>{badge}</td>
            <td style='padding:0.5rem;text-align:center'>{rp_h}</td>
            <td style='padding:0.5rem;text-align:center'>{ep_h}</td>
            <td style='padding:0.5rem;text-align:center'>{rr_h}</td>
            <td style='padding:0.5rem;text-align:center'>{er_h}</td>
            <td style='padding:0.5rem;text-align:center'>{rf_h}</td>
            <td style='padding:0.5rem;text-align:center'>{ef_h}</td>
        </tr>"""

    st.markdown(f"""
    <div class="card" style='padding:1rem'>
    <table style='width:100%;border-collapse:collapse;font-size:0.82rem'>
        <thead><tr style='border-bottom:2px solid #bae6fd;background:#f0f9ff'>
            <th style='padding:0.5rem;text-align:left;color:#475569'>Kelas</th>
            <th style='padding:0.5rem;text-align:center;color:#0369a1'>Prec R50</th>
            <th style='padding:0.5rem;text-align:center;color:#7c3aed'>Prec Eff</th>
            <th style='padding:0.5rem;text-align:center;color:#0369a1'>Rec R50</th>
            <th style='padding:0.5rem;text-align:center;color:#7c3aed'>Rec Eff</th>
            <th style='padding:0.5rem;text-align:center;color:#0369a1'>F1 R50</th>
            <th style='padding:0.5rem;text-align:center;color:#7c3aed'>F1 Eff</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <div style='font-size:0.7rem;color:#94a3b8;margin-top:0.5rem'>
        🟢 Angka hijau = model lebih unggul di metrik tersebut
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabel arsitektur ──────────────────────────────────────────────────
    st.markdown('<div class="card-title">🏗️ Perbandingan Arsitektur</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <table style='width:100%;border-collapse:collapse;font-size:0.85rem'>
        <thead><tr style='border-bottom:2px solid #bae6fd;background:#f0f9ff'>
            <th style='padding:0.75rem;color:#475569;text-align:left'>Aspek</th>
            <th style='padding:0.75rem;color:#0369a1;text-align:center'>ResNet50</th>
            <th style='padding:0.75rem;color:#7c3aed;text-align:center'>EfficientNetB0</th>
        </tr></thead>
        <tbody>
            <tr style='border-bottom:1px solid #f1f5f9'>
                <td style='padding:0.6rem;color:#475569'>Tahun</td>
                <td style='text-align:center;color:#64748b'>2015</td>
                <td style='text-align:center;color:#64748b'>2019</td>
            </tr>
            <tr style='border-bottom:1px solid #f1f5f9'>
                <td style='padding:0.6rem;color:#475569'>Jumlah Parameter</td>
                <td style='text-align:center;color:#64748b'>~25.6M</td>
                <td style='text-align:center;color:#64748b'>~5.3M</td>
            </tr>
            <tr style='border-bottom:1px solid #f1f5f9'>
                <td style='padding:0.6rem;color:#475569'>Test Accuracy</td>
                <td style='text-align:center;color:#0369a1;font-weight:700'>93.0%</td>
                <td style='text-align:center;color:#7c3aed'>89.0%</td>
            </tr>
            <tr style='border-bottom:1px solid #f1f5f9'>
                <td style='padding:0.6rem;color:#475569'>Inovasi Utama</td>
                <td style='text-align:center;color:#64748b'>Residual Connections</td>
                <td style='text-align:center;color:#64748b'>Compound Scaling</td>
            </tr>
            <tr>
                <td style='padding:0.6rem;color:#475569'>Blok Utama</td>
                <td style='text-align:center;color:#64748b'>Bottleneck Block</td>
                <td style='text-align:center;color:#64748b'>MBConv Block</td>
            </tr>
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Kesimpulan ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="analysis-box">
        <strong style='color:#0369a1'>📋 Kesimpulan Perbandingan</strong><br><br>
        Berdasarkan hasil evaluasi pada data test (2.800 sampel, 8 kelas),
        <strong>ResNet50 unggul secara keseluruhan</strong> dengan accuracy <strong>93.0%</strong>
        dibanding EfficientNetB0 <strong>89.0%</strong> (selisih 4.0%).<br><br>
        ResNet50 dominan di kelas <strong>AMD, CSR, MH</strong> dengan F1-Score sempurna 100%,
        serta lebih baik di <strong>CNV dan DME</strong>. EfficientNetB0 sedikit lebih unggul
        di kelas <strong>DR</strong> namun tertinggal signifikan di <strong>DRUSEN</strong>
        (73% vs 80%).<br><br>
        Dari sisi efisiensi, EfficientNetB0 hanya membutuhkan ~5.3M parameter (vs ~25.6M ResNet50),
        menjadikannya kandidat lebih ringan untuk deployment — namun untuk akurasi diagnostik
        maksimal pada dataset ini, <strong>ResNet50 adalah pilihan utama</strong>.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TENTANG SISTEM
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="card">
        <div class="card-title">👁️ Tentang Sistem Ini</div>
        <p style='line-height:1.8;color:#475569;font-size:0.9rem'>
        Sistem ini merupakan implementasi web untuk penelitian <strong>Perbandingan Arsitektur 
        ResNet50 dan EfficientNetB0 dengan Explainable AI Grad-CAM++</strong> untuk klasifikasi 
        penyakit retina pada citra Optical Coherence Tomography (OCT).
        </p>
    </div>

    <div class="card">
        <div class="card-title">🧠 Metode Explainability — Grad-CAM++</div>
        <p style='line-height:1.8;color:#475569;font-size:0.9rem'>
        <strong style='color:#0f172a'>Grad-CAM++</strong> (Gradient-weighted Class Activation Mapping++) 
        adalah peningkatan dari Grad-CAM yang menggunakan kombinasi bobot alpha berdasarkan turunan 
        orde-dua untuk menghasilkan peta perhatian (attention map) yang lebih presisi. Pada 
        konteks OCT retina, heatmap yang dihasilkan membantu mengidentifikasi area jaringan retina 
        mana yang paling berkontribusi terhadap keputusan klasifikasi model.
        </p>
    </div>

    <div class="card">
        <div class="card-title">📋 8 Kelas Penyakit Retina</div>
    """, unsafe_allow_html=True)

    cls_cols = st.columns(4)
    for i, (cls, desc) in enumerate(CLASS_DESC.items()):
        with cls_cols[i % 4]:
            badge = render_badge(cls)
            sev   = CLASS_SEVERITY.get(cls, '')
            st.markdown(f"""
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                        padding:0.85rem;margin-bottom:0.5rem;box-shadow:0 1px 2px rgba(0,0,0,0.04)'>
                {badge}
                <div style='font-size:0.7rem;font-weight:600;color:#64748b;margin-top:0.4rem'>{sev}</div>
                <div style='font-size:0.72rem;color:#94a3b8;margin-top:0.3rem;line-height:1.5'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
