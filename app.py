"""
app.py
======
College Project: Vehicle Number Plate Image Enhancement and Sharpening System
Clean, Production-Ready Streamlit Web Application for Cloud Deployment
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
from pathlib import Path
import json
import io
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Add workspace root to sys.path for cross-platform imports
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.enhancement.blur_detector import detect_blur
from src.enhancement.degradation import (
    apply_motion_blur,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_contrast
)
from src.enhancement.noise_reduction import (
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_median_filter
)
from src.enhancement.sharpening import (
    laplacian_sharpen,
    highpass_sharpen,
    unsharp_mask,
    adaptive_plate_enhance
)
from src.ocr.reader import recognize_plate_text
from src.evaluation.metrics import evaluate_image_quality, compute_ocr_accuracy
from src.evaluation.evaluator import evaluate_single_plate


# ----------------------------------------------------
# PAGE CONFIGURATION & METADATA
# ----------------------------------------------------
st.set_page_config(
    page_title="Vehicle Plate Enhancement AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# MODERN UI STYLING (CSS)
# ----------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top banner styling */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.35);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric card design */
    .metric-card {
        background: #161E2E;
        border: 1px solid #283548;
        border-radius: 12px;
        padding: 16px 18px;
        text-align: left;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #3B82F6;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .metric-delta-pos {
        font-size: 0.85rem;
        font-weight: 600;
        color: #34D399;
        margin-top: 4px;
    }
    .metric-delta-neg {
        font-size: 0.85rem;
        font-weight: 600;
        color: #F87171;
        margin-top: 4px;
    }

    /* Badges */
    .status-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pill-red {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .pill-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Clean Card Container */
    .panel-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
    }

    /* Streamlit UI elements polish */
    div[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# CACHED DATA LOADERS & OCR INFERENCE
# ----------------------------------------------------
@st.cache_data
def load_sample_plates_list():
    """Loads sample plates index safely across Windows & Linux environments."""
    index_path = Path("data/sample_plates/samples_index.json")
    if not index_path.exists():
        index_path = workspace_root / "data" / "sample_plates" / "samples_index.json"

    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


@st.cache_data(show_spinner=False, max_entries=80)
def cached_recognize_ocr(image_bytes: bytes) -> dict:
    """
    Caches OCR output based on image bytes to avoid slow deep-learning
    re-inference every time interactive UI sliders are adjusted.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"text": "", "raw_text": "", "confidence": 0.0, "is_valid_format": False}
    try:
        return recognize_plate_text(img)
    except Exception as e:
        return {
            "text": "",
            "raw_text": f"Error: {e}",
            "confidence": 0.0,
            "is_valid_format": False
        }


def encode_image_bytes(image: np.ndarray, ext: str = ".png") -> bytes:
    """Encodes an OpenCV image to in-memory bytes."""
    success, buffer = cv2.imencode(ext, image)
    return buffer.tobytes() if success else b""


# ----------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### 🚗 License Plate Source")
    input_source = st.radio(
        "Choose Source Mode",
        ["Pre-loaded Dataset Plates", "Upload Custom Plate Image"],
        label_visibility="collapsed"
    )

    current_image = None
    ground_truth = None
    image_title = ""

    if input_source == "Pre-loaded Dataset Plates":
        samples = load_sample_plates_list()
        if samples:
            sample_labels = [f"🇮🇳 {s['ground_truth']} ({s['filename'][:20]}...)" for s in samples]
            selected_idx = st.selectbox(
                "Select Plate Sample",
                range(len(samples)),
                format_func=lambda i: sample_labels[i]
            )
            selected_sample = samples[selected_idx]
            ground_truth = selected_sample.get("ground_truth", "")
            image_title = f"Plate: {ground_truth}"

            # Cross-platform safe path resolution
            possible_path = Path("data/sample_plates") / selected_sample["filename"]
            if not possible_path.exists():
                possible_path = workspace_root / "data" / "sample_plates" / selected_sample["filename"]
            if not possible_path.exists() and "path" in selected_sample:
                possible_path = Path(selected_sample["path"].replace("\\", "/"))

            if possible_path.exists():
                current_image = cv2.imread(str(possible_path))
        else:
            st.warning("⚠️ No sample plates found in data/sample_plates/")

    else:
        uploaded_file = st.file_uploader(
            "Upload Vehicle / Plate Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            current_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_title = uploaded_file.name
            gt_input = st.text_input("Ground Truth Number (Optional):", placeholder="e.g. MH12DE1433")
            if gt_input.strip():
                ground_truth = gt_input.strip().upper()

    st.markdown("---")

    # Degradation Simulator Accordion
    with st.expander("🌪️ Camera Flaw Simulator", expanded=False):
        st.caption("Inject synthetic optics or motion degradation to test restoration:")
        simulate_deg = st.checkbox("Simulate Camera Degradation", value=False)
        deg_type = "None"
        streak = 11
        ksize = 7
        sigma_noise = 20.0
        sp_amt = 0.03
        contrast_factor = 0.5

        if simulate_deg:
            deg_type = st.selectbox(
                "Degradation Artifact",
                [
                    "Motion Blur (Vehicle Velocity)",
                    "Defocus Blur (Out of Focus)",
                    "Sensor Grain (Gaussian Noise)",
                    "Impulse (Salt & Pepper)",
                    "Low Illumination / Night Contrast"
                ]
            )
            if "Motion" in deg_type:
                streak = st.slider("Streak Length (px)", 3, 31, 13, step=2)
            elif "Defocus" in deg_type:
                ksize = st.slider("Gaussian Blur Kernel", 3, 21, 7, step=2)
            elif "Sensor" in deg_type:
                sigma_noise = st.slider("Noise Standard Deviation", 5.0, 50.0, 22.0)
            elif "Impulse" in deg_type:
                sp_amt = st.slider("Noise Density", 0.01, 0.12, 0.03, step=0.01)
            elif "Low Illumination" in deg_type:
                contrast_factor = st.slider("Contrast Factor", 0.15, 0.90, 0.45, step=0.05)

    st.markdown("---")

    # Enhancement Settings
    st.markdown("### 🛠️ Enhancement Settings")
    enhancement_mode = st.selectbox(
        "Enhancement Technique",
        [
            "Adaptive High-Contrast (CLAHE + USM) ⭐",
            "Unsharp Masking (USM)",
            "Laplacian 2nd-Derivative Sharpening",
            "Spatial High-Pass Filter"
        ]
    )

    with st.expander("⚙️ Fine-Tune Parameters", expanded=True):
        use_bilateral = st.checkbox(
            "Apply Edge-Preserving Denoising",
            value=False,
            help="Enable if image contains sensor noise or compression grain."
        )
        if use_bilateral:
            b_diameter = st.slider("Bilateral Diameter", 3, 13, 5, step=2)
            b_sigma = st.slider("Bilateral Color Sigma", 15, 90, 45)

        if "Adaptive" in enhancement_mode:
            clahe_clip = st.slider("CLAHE Contrast Boost", 1.0, 6.0, 3.2, step=0.2)
            usm_amt = st.slider("Sharpening Strength", 1.0, 4.5, 2.4, step=0.2)
        elif "Unsharp" in enhancement_mode:
            usm_amt = st.slider("USM Amount", 0.5, 4.5, 2.2, step=0.1)
            usm_thresh = st.slider("Threshold (Noise suppression)", 0, 10, 1)
        elif "Laplacian" in enhancement_mode:
            lap_alpha = st.slider("Laplacian Weight (α)", 0.2, 2.5, 0.8, step=0.1)
        else:
            hpf_beta = st.slider("High-Pass Boost (β)", 0.2, 3.0, 1.1, step=0.1)

    st.markdown("---")

    # OCR Control Mode
    st.markdown("### ⚡ Inference Engine")
    auto_run_ocr = st.checkbox("Auto-Run OCR on Adjustment", value=True, help="Uses smart caching for real-time responsiveness.")
    st.caption("Deployment: Cloud CPU-Optimized • EasyOCR Engine")


# ----------------------------------------------------
# MAIN DASHBOARD HEADER
# ----------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Deep Computer Vision • Image Sharpening System</div>
    <div class="hero-title">Vehicle Number Plate Enhancement & Recognition</div>
    <div class="hero-subtitle">
        Interactive classical computer vision framework for camera degradation restoration, edge sharpening, and high-accuracy OCR recognition.
    </div>
</div>
""", unsafe_allow_html=True)

if current_image is None:
    st.info("👈 Please select a sample plate or upload a vehicle image from the sidebar to begin.")
    st.stop()


# ----------------------------------------------------
# PROCESSING PIPELINE EXECUTION
# ----------------------------------------------------
# Step 1: Synthesize degradation if requested
if simulate_deg and current_image is not None:
    if "Motion" in deg_type:
        processed_input = apply_motion_blur(current_image, length=streak, angle=15.0)
    elif "Defocus" in deg_type:
        processed_input = apply_gaussian_blur(current_image, kernel_size=ksize, sigma=2.0)
    elif "Sensor" in deg_type:
        processed_input = apply_gaussian_noise(current_image, sigma=sigma_noise)
    elif "Impulse" in deg_type:
        processed_input = apply_salt_and_pepper_noise(current_image, amount=sp_amt)
    elif "Low Illumination" in deg_type:
        processed_input = apply_low_contrast(current_image, factor=contrast_factor, brightness_bias=-20)
    else:
        processed_input = current_image.copy()
else:
    processed_input = current_image.copy()

# Step 2: Noise reduction
if use_bilateral:
    stage1 = apply_bilateral_filter(processed_input, d=b_diameter, sigma_color=b_sigma, sigma_space=b_sigma)
else:
    stage1 = processed_input.copy()

# Step 3: Sharpening
if "Adaptive" in enhancement_mode:
    enhanced = adaptive_plate_enhance(stage1, clahe_clip=clahe_clip, usm_amount=usm_amt, denoise=False)
elif "Unsharp" in enhancement_mode:
    enhanced = unsharp_mask(stage1, amount=usm_amt, threshold=usm_thresh)
elif "Laplacian" in enhancement_mode:
    enhanced = laplacian_sharpen(stage1, alpha=lap_alpha)
else:
    enhanced = highpass_sharpen(stage1, beta=hpf_beta)


# ----------------------------------------------------
# TABS NAVIGATION
# ----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Live Pipeline & Diagnostics",
    "⚖️ Sharpening Matrix Comparison",
    "🧪 Dataset Benchmark",
    "📚 Mathematical Foundations & Viva"
])


# ====================================================
# TAB 1: LIVE PIPELINE & DIAGNOSTICS
# ====================================================
with tab1:
    # 1. Blur Diagnostics
    raw_blur = detect_blur(processed_input)
    enh_blur = detect_blur(enhanced)
    sharp_boost = enh_blur['laplacian_var'] - raw_blur['laplacian_var']
    tenengrad_boost = enh_blur['tenengrad_score'] - raw_blur['tenengrad_score']

    # 2. OCR Inference (Cached)
    raw_bytes = encode_image_bytes(processed_input)
    enh_bytes = encode_image_bytes(enhanced)

    if auto_run_ocr:
        with st.spinner("Extracting text via OCR..."):
            raw_ocr = cached_recognize_ocr(raw_bytes)
            enh_ocr = cached_recognize_ocr(enh_bytes)
    else:
        if st.button("🚀 Run OCR Analysis on Current Plate", type="primary"):
            with st.spinner("Extracting text via OCR..."):
                raw_ocr = cached_recognize_ocr(raw_bytes)
                enh_ocr = cached_recognize_ocr(enh_bytes)
        else:
            raw_ocr = {"text": "Click 'Run OCR'", "confidence": 0.0, "is_valid_format": False}
            enh_ocr = {"text": "Click 'Run OCR'", "confidence": 0.0, "is_valid_format": False}

    # 3. Quality Metrics
    quality = evaluate_image_quality(enhanced, reference=current_image)
    conf_diff = enh_ocr.get('confidence', 0.0) - raw_ocr.get('confidence', 0.0)

    # 4. Top Metric Bar
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PSNR (Fidelity)</div>
            <div class="metric-value">{quality.get('psnr_db', 0):.1f} dB</div>
            <div class="metric-delta-pos">Peak Signal-to-Noise</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SSIM Index</div>
            <div class="metric-value">{quality.get('ssim', 0):.3f}</div>
            <div class="metric-delta-pos">Structure Retention</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CNR (Contrast/Noise)</div>
            <div class="metric-value">{quality.get('cnr', 0):.2f}</div>
            <div class="metric-delta-pos">Character Edge Contrast</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        delta_class = "metric-delta-pos" if sharp_boost >= 0 else "metric-delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Laplacian Var</div>
            <div class="metric-value">{enh_blur['laplacian_var']:.0f}</div>
            <div class="{delta_class}">{sharp_boost:+.0f} Sharpness Delta</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        delta_class = "metric-delta-pos" if conf_diff >= 0 else "metric-delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">OCR Confidence</div>
            <div class="metric-value">{enh_ocr.get('confidence', 0.0):.2f}</div>
            <div class="{delta_class}">{conf_diff:+.2f} vs Raw Plate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Side-by-Side Images
    col_raw, col_enh = st.columns(2)

    with col_raw:
        st.markdown("#### 1️⃣ Input / Degraded Plate")
        st.image(cv2.cvtColor(processed_input, cv2.COLOR_BGR2RGB), use_container_width=True)
        raw_text_display = raw_ocr.get("text") or raw_ocr.get("raw_text") or "[No Text Detected]"
        pill_status = '<span class="status-pill pill-red">🔴 Blurry</span>' if raw_blur['is_blurry'] else '<span class="status-pill pill-green">🟢 Sharp</span>'
        st.markdown(f"""
        - **Sharpness Status:** {pill_status}
        - **Laplacian Variance:** `{raw_blur['laplacian_var']:.1f}`
        - **Tenengrad Energy:** `{raw_blur['tenengrad_score']:.1f}`
        - **Detected Text:** `{raw_text_display}`
        - **OCR Confidence:** `{raw_ocr.get('confidence', 0.0):.2f}`
        """)

    with col_enh:
        st.markdown("#### 2️⃣ Enhanced & Restored Plate")
        st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_container_width=True)
        enh_text_display = enh_ocr.get("text") or enh_ocr.get("raw_text") or "[No Text Detected]"
        enh_status = '<span class="status-pill pill-green">🟢 Enhanced & Sharp</span>' if not enh_blur['is_blurry'] else '<span class="status-pill pill-amber">⚠️ Partial Blur</span>'
        valid_pill = '<span class="status-pill pill-blue">🇮🇳 Valid Indian Format</span>' if enh_ocr.get('is_valid_format') else '<span class="status-pill pill-amber">Standard Format</span>'
        st.markdown(f"""
        - **Sharpness Status:** {enh_status} {valid_pill}
        - **Laplacian Variance:** `{enh_blur['laplacian_var']:.1f}` (**+{sharp_boost:.1f}**)
        - **Tenengrad Energy:** `{enh_blur['tenengrad_score']:.1f}` (**+{tenengrad_boost:.1f}**)
        - **Detected Text:** `{enh_text_display}`
        - **OCR Confidence:** `{enh_ocr.get('confidence', 0.0):.2f}`
        """)

    # Download Button
    enh_png = encode_image_bytes(enhanced, ".png")
    st.download_button(
        label="💾 Download Enhanced Plate (PNG)",
        data=enh_png,
        file_name="enhanced_number_plate.png",
        mime="image/png"
    )

    # 6. Ground Truth Verification Card (if available)
    if ground_truth:
        st.markdown("---")
        gt_acc = compute_ocr_accuracy(ground_truth, enh_ocr.get('text', ''))
        st.markdown(f"""
        <div class="panel-card">
            <h4 style="margin: 0 0 10px 0; color: #60A5FA;">🎯 Ground Truth Evaluation</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: center;">
                <div><strong>Ground Truth:</strong> <code style="font-size: 1.1rem; color: #F8FAFC;">{ground_truth}</code></div>
                <div><strong>Enhanced OCR:</strong> <code style="font-size: 1.1rem; color: #34D399;">{enh_ocr.get('text', '') or 'N/A'}</code></div>
                <div><strong>Match Status:</strong> {'<span class="status-pill pill-green">✅ EXACT MATCH</span>' if gt_acc['exact_match'] else '<span class="status-pill pill-red">❌ MISMATCH</span>'}</div>
                <div><strong>Character Error Rate (CER):</strong> <code>{gt_acc['cer']:.3f}</code></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 7. Advanced Visual Analysis Tools
    with st.expander("🔍 Advanced Inspection: Edge Gradient, Binarization & Intensity Profile", expanded=False):
        tool_choice = st.radio(
            "Select Diagnostic Visualization",
            ["High-Frequency Residual Map", "Character Stroke Binarization (Otsu)", "Pixel Intensity Cross-Section Curve"],
            horizontal=True
        )

        if tool_choice == "High-Frequency Residual Map":
            st.caption("Visualizing high-frequency edge gradients boosted by the enhancement filter (Enhanced - Raw residual):")
            # Difference map amplified
            g_raw = cv2.cvtColor(processed_input, cv2.COLOR_BGR2GRAY) if len(processed_input.shape) == 3 else processed_input
            g_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            diff = cv2.absdiff(g_enh, g_raw)
            diff_color = cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=3.0), cv2.COLORMAP_JET)
            st.image(cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB), caption="High-Frequency Edge Magnification (Jet Heatmap)", use_container_width=True)

        elif tool_choice == "Character Stroke Binarization (Otsu)":
            b1, b2 = st.columns(2)
            g_raw = cv2.cvtColor(processed_input, cv2.COLOR_BGR2GRAY) if len(processed_input.shape) == 3 else processed_input
            g_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            _, bin_raw = cv2.threshold(g_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, bin_enh = cv2.threshold(g_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            with b1:
                st.image(bin_raw, caption="Raw Input Plate (Binarized)", use_container_width=True)
            with b2:
                st.image(bin_enh, caption="Enhanced Plate (Binarized Character Strokes)", use_container_width=True)

        else:
            st.caption("Cross-section pixel intensity profile along the middle row of characters (steeper transition = sharper text stroke):")
            g_raw = cv2.cvtColor(processed_input, cv2.COLOR_BGR2GRAY) if len(processed_input.shape) == 3 else processed_input
            g_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            mid_row = g_raw.shape[0] // 2
            row_raw = g_raw[mid_row, :]
            row_enh = g_enh[mid_row, :]

            fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#111827")
            ax.set_facecolor("#161E2E")
            ax.plot(row_raw, label="Raw Input Profile", color="#F87171", linewidth=1.5, linestyle="--")
            ax.plot(row_enh, label="Enhanced Profile", color="#34D399", linewidth=2.0)
            ax.set_xlabel("Pixel Horizontal Index (X)", color="#94A3B8", fontsize=9)
            ax.set_ylabel("Grayscale Intensity (0-255)", color="#94A3B8", fontsize=9)
            ax.tick_params(colors="#94A3B8")
            ax.grid(True, color="#283548", linestyle=":", alpha=0.6)
            ax.legend(facecolor="#161E2E", edgecolor="#283548", labelcolor="#F8FAFC")
            st.pyplot(fig)
            plt.close(fig)


# ====================================================
# TAB 2: SHARPENING MATRIX COMPARISON
# ====================================================
with tab2:
    st.markdown("### ⚖️ Side-by-Side Sharpening Operator Evaluation")
    st.write("Compare the spatial and frequency response of all four core algorithms on the active plate:")

    c_lap = laplacian_sharpen(stage1, alpha=0.8)
    c_hpf = highpass_sharpen(stage1, beta=1.0)
    c_usm = unsharp_mask(stage1, amount=2.0, threshold=1)
    c_ada = adaptive_plate_enhance(stage1, clahe_clip=3.0, usm_amount=2.5)

    col1, col2, col3, col4 = st.columns(4)

    methods_data = [
        ("Laplacian (2nd Deriv.)", c_lap, col1),
        ("Spatial High-Pass", c_hpf, col2),
        ("Unsharp Mask (USM)", c_usm, col3),
        ("Adaptive High-Contrast", c_ada, col4)
    ]

    metrics_rows = []

    for name, img, col in methods_data:
        b_info = detect_blur(img)
        q_info = evaluate_image_quality(img, reference=current_image)
        with col:
            st.markdown(f"**{name}**")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Sharpness: **{b_info['laplacian_var']:.1f}**\n\nCNR: **{q_info['cnr']:.2f}**")
        metrics_rows.append({
            "Enhancement Technique": name,
            "Laplacian Variance": round(b_info["laplacian_var"], 1),
            "Tenengrad Score": round(b_info["tenengrad_score"], 1),
            "CNR": round(q_info["cnr"], 2),
            "PSNR (dB)": round(q_info.get("psnr_db", 0), 1),
            "SSIM": round(q_info.get("ssim", 0), 4)
        })

    st.markdown("#### 📊 Quantitative Matrix")
    st.dataframe(metrics_rows, use_container_width=True)

    st.markdown("""
    <div class="panel-card">
        <h5 style="color: #60A5FA; margin-top: 0;">🎓 Academic Method Analysis:</h5>
        <ul style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 0;">
            <li><strong>Laplacian Sharpening:</strong> Enhances rapid spatial transitions via 2nd-order derivatives; excellent for crisp plates but amplifies sensor grain if noise is unsuppressed.</li>
            <li><strong>High-Pass Filter:</strong> Subtracts low-pass components from original, isolating high-frequency boundary information.</li>
            <li><strong>Unsharp Masking (USM):</strong> Edge-enhancement standard from photography; produces clean visual borders with minimal overshoot.</li>
            <li><strong>Adaptive High-Contrast (Recommended):</strong> Combines CLAHE local histogram normalization with USM high-frequency boost, giving the highest OCR recognition gains under extreme blur or harsh shadows.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ====================================================
# TAB 3: DATASET BENCHMARK
# ====================================================
with tab3:
    st.markdown("### 🧪 Pre-loaded Indian Plates Dataset Benchmark")
    samples = load_sample_plates_list()

    if samples:
        st.write(f"Total curated plates in demonstration dataset: **{len(samples)}**")

        table_rows = []
        for s in samples:
            table_rows.append({
                "Filename": s.get("filename", ""),
                "Ground Truth": s.get("ground_truth", ""),
                "Source Vehicle": s.get("original_image", ""),
                "Width (px)": s.get("dimensions", [0, 0])[0],
                "Height (px)": s.get("dimensions", [0, 0])[1]
            })
        st.dataframe(table_rows, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🚀 Run Automated Batch Evaluation")
        batch_size = st.slider("Select Sample Size to Benchmark", 3, min(len(samples), 15), 5)

        if st.button("Run Batch Evaluation", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            benchmark_results = []

            for idx, s in enumerate(samples[:batch_size]):
                status_text.text(f"Evaluating plate {idx + 1}/{batch_size}: {s.get('ground_truth', '')}...")
                p_path = Path("data/sample_plates") / s["filename"]
                if not p_path.exists():
                    p_path = workspace_root / "data" / "sample_plates" / s["filename"]

                if p_path.exists():
                    p_img = cv2.imread(str(p_path))
                    if p_img is not None:
                        eval_dict = evaluate_single_plate(p_img, ground_truth_text=s.get("ground_truth", ""))
                        raw = eval_dict.get("Raw Input", {})
                        hybrid = eval_dict.get("Bilateral + USM (Hybrid)", {})

                        benchmark_results.append({
                            "Plate": s.get("ground_truth", ""),
                            "Raw Sharpness": raw.get("laplacian_var", 0),
                            "Enhanced Sharpness": hybrid.get("laplacian_var", 0),
                            "Raw OCR": raw.get("ocr_text", "") or "[None]",
                            "Enhanced OCR": hybrid.get("ocr_text", "") or "[None]",
                            "Exact Match": "✅ YES" if hybrid.get("exact_match") else "❌ NO",
                            "CER": round(hybrid.get("cer", 0.0), 3)
                        })
                progress_bar.progress((idx + 1) / batch_size)

            status_text.text("✅ Benchmark complete!")
            st.dataframe(benchmark_results, use_container_width=True)

            # Summary stats
            exact_count = sum(1 for r in benchmark_results if r["Exact Match"] == "✅ YES")
            accuracy_pct = (exact_count / len(benchmark_results)) * 100 if benchmark_results else 0
            avg_cer = np.mean([r["CER"] for r in benchmark_results]) if benchmark_results else 0

            s1, s2, s3 = st.columns(3)
            s1.metric("Batch Recognition Accuracy", f"{accuracy_pct:.1f}%", f"{exact_count}/{len(benchmark_results)} plates")
            s2.metric("Average Character Error Rate", f"{avg_cer:.3f}")
            s3.metric("Plates Tested", f"{len(benchmark_results)}")
    else:
        st.warning("⚠️ No dataset samples found in data/sample_plates/")


# ====================================================
# TAB 4: MATHEMATICAL FOUNDATIONS & VIVA PREP
# ====================================================
with tab4:
    st.markdown("### 📚 Mathematical Foundations & Viva Voce Preparation")

    st.markdown(r"""
    #### 1. Blur Detection: Variance of the Laplacian
    The discrete Laplacian operator approximates the second spatial derivative of an image $I(x, y)$:
    $$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
    The $3 \times 3$ isotropic convolution kernel used in our system:
    $$\mathbf{K}_{\text{Laplacian}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
    The sharpness score $S$ is quantified by the statistical variance $\sigma^2$:
    $$S(I) = \text{Var}(\nabla^2 I) = \frac{1}{N} \sum_{x, y} \left( (\nabla^2 I)(x, y) - \mu_{\nabla^2 I} \right)^2$$
    - **Physical intuition:** Rapid intensity transitions produce high response spikes, resulting in high variance.
    - **Blur degradation:** Spreads edge slopes, reducing derivative amplitudes and yielding low variance.

    ---

    #### 2. Edge-Preserving Denoising: Bilateral Filter
    Unlike linear Gaussian smoothing which blurs across edges, the bilateral filter combines geometric domain distance and pixel radiometric intensity similarity:
    $$BF[I]_p = \frac{1}{W_p} \sum_{q \in S} \exp\left(-\frac{\|p - q\|^2}{2\sigma_s^2}\right) \exp\left(-\frac{|I_p - I_q|^2}{2\sigma_r^2}\right) I_q$$
    Where:
    - $\sigma_s$ controls spatial proximity weighting.
    - $\sigma_r$ controls intensity difference tolerance (preserves sharp character strokes on plate backgrounds).

    ---

    #### 3. High-Frequency Sharpening: Unsharp Masking (USM)
    High frequencies are amplified by subtracting a Gaussian low-pass smoothed mask from the original signal:
    $$\text{Mask}(x, y) = I(x, y) - (I * G_\sigma)(x, y)$$
    $$I_{\text{sharpened}}(x, y) = I(x, y) + k \cdot \text{Mask}(x, y)$$
    Where $k$ is the scaling strength factor (typically $1.5$ to $2.5$).

    ---

    #### 4. Image Quality Metrics
    - **Peak Signal-to-Noise Ratio (PSNR):**
      $$\text{PSNR} = 20 \log_{10}\left(\frac{\text{MAX}_I}{\sqrt{\text{MSE}}}\right) = 20 \log_{10}\left(\frac{255}{\sqrt{\frac{1}{MN}\sum (I - \hat{I})^2}}\right)$$
    - **Structural Similarity Index (SSIM):**
      $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
    - **Character Error Rate (CER):**
      $$\text{CER} = \frac{\text{Levenshtein Edit Distance}(T_{\text{ground\_truth}}, T_{\text{OCR}})}{\text{Length}(T_{\text{ground\_truth}})}$$
    """)

    st.markdown("---")
    st.markdown("#### 🎯 Examiner Viva Voce FAQs")

    with st.expander("Q1: Why use classical computer vision instead of an end-to-end deep learning model?", expanded=False):
        st.write("""
        **Answer:** Classical digital image processing (Laplacian, Unsharp Masking, Bilateral Filtering) operates deterministically in real-time with sub-10ms latency on edge hardware (e.g. traffic cameras, Raspberry Pi, low-power edge gateways) without requiring expensive GPU compute or massive labelled training data.
        """)

    with st.expander("Q2: Why not just use standard Gaussian blur before sharpening?", expanded=False):
        st.write("""
        **Answer:** Standard Gaussian smoothing blurs across character boundaries equally, destroying thin alphanumeric strokes. The Bilateral filter uses both spatial and range Gaussian kernels to smooth intra-region sensor noise while completely preserving the sharp contrast between black characters and the white/yellow plate background.
        """)

    with st.expander("Q3: How does CLAHE enhance OCR recognition on poorly illuminated plates?", expanded=False):
        st.write("""
        **Answer:** Contrast Limited Adaptive Histogram Equalization operates on local $8 \times 8$ tiles rather than the global image. It normalizes uneven illumination caused by car headlights or shadows while the contrast clip limit prevents noise amplification in flat areas.
        """)
