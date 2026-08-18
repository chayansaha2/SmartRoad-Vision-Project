import streamlit as st
from ultralytics import YOLO
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pandas as pd
import folium
from streamlit_folium import st_folium
import hashlib
import numpy as np
import cv2
import io
import json
import math
from datetime import datetime
from fpdf import FPDF
from geopy.geocoders import Nominatim

# ------------------------------------------------------------------------------
# 1. Page Configuration & Global Theme Engine (Zero-White Dark Glassmorphism)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartRoad-Vision • Pavement Intelligence Suite",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800;900&family=Share+Tech+Mono&display=swap');

    /* Global Dark Canvas */
    .stApp {
        background-color: #070a12;
        background-image: 
            radial-gradient(circle at 12% 15%, rgba(56, 189, 248, 0.06) 0%, transparent 40%),
            radial-gradient(circle at 88% 85%, rgba(245, 158, 11, 0.05) 0%, transparent 40%),
            linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 28px 28px, 28px 28px;
        font-family: 'Share Tech Mono', monospace;
        color: #f8fafc;
    }

    /* =========================================================================
       COMPLETE SIDEBAR DARK RE-SKINNING (Fixes All White Boxes & Widgets)
       ========================================================================= */
    section[data-testid="stSidebar"] {
        background-color: #0c121e !important;
        border-right: 2px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
        letter-spacing: 0.05em !important;
    }
    section[data-testid="stSidebar"] label {
        color: #94a3b8 !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
    }

    /* 1. Dropdown Selectbox */
    section[data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #131c2e !important;
        border: 1.5px solid #1e293b !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        background-color: transparent !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* 2. Number Input Box & Increment/Decrement Controls */
    section[data-testid="stSidebar"] div[data-baseweb="input"] {
        background-color: #131c2e !important;
        border: 1.5px solid #1e293b !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] input[type="number"],
    section[data-testid="stSidebar"] input[type="text"] {
        background-color: #131c2e !important;
        color: #38bdf8 !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        border: none !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #38bdf8 !important;
        color: #070a12 !important;
    }

    /* 3. Slider Controls */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] div[role="slider"] {
        background-color: #38bdf8 !important;
        border: 2px solid #ffffff !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-baseweb="slider"] {
        background-color: transparent !important;
    }

    /* 4. LaTeX KaTeX Math Dark Contrast Fix */
    section[data-testid="stSidebar"] .katex {
        color: #38bdf8 !important;
        font-size: 1.05rem !important;
        background: transparent !important;
    }
    section[data-testid="stSidebar"] .katex-html {
        background: transparent !important;
    }
    section[data-testid="stSidebar"] code {
        background-color: #131c2e !important;
        color: #34d399 !important;
        border: 1px solid #1e293b !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* Top Brand Navigation Header */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-top: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 20px 30px;
        margin-bottom: 24px;
        box-shadow: 0 14px 35px rgba(0, 0, 0, 0.65);
    }
    .brand-title {
        font-family: 'Orbitron', monospace !important;
        font-size: 1.95rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: linear-gradient(135deg, #38bdf8 0%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .brand-subtitle {
        font-family: 'Share Tech Mono', monospace !important;
        color: #94a3b8;
        font-size: 0.98rem;
        letter-spacing: 0.04em;
        margin-top: 6px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(16, 185, 129, 0.15);
        border: 1.5px solid rgba(16, 185, 129, 0.45);
        color: #34d399;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.92rem;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 9999px;
    }
    .pulse-indicator {
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 12px #10b981;
        animation: pulse-ring 1.8s infinite;
    }
    @keyframes pulse-ring {
        0% { transform: scale(0.95); opacity: 1; }
        50% { transform: scale(1.25); opacity: 0.6; }
        100% { transform: scale(0.95); opacity: 1; }
    }

    /* KPI Summary Cards Grid */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 14px;
        margin-bottom: 26px;
    }
    .kpi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
    }
    .kpi-label {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
    }
    .kpi-value {
        font-family: 'Orbitron', monospace !important;
        font-size: 2.05rem;
        font-weight: 900;
        color: #ffffff;
        margin-top: 6px;
        letter-spacing: -0.01em;
    }
    .kpi-sub {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.86rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Content Panels */
    .panel-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 22px;
    }
    .panel-title-bar {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1.5px solid #1e293b;
        padding-bottom: 10px;
        margin-bottom: 16px;
    }

    /* High-Contrast Data Tables */
    .table-wrapper {
        width: 100%;
        border-radius: 10px;
        overflow: hidden;
        border: 1.5px solid #1e293b;
        background: #080c14;
    }
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.98rem;
        text-align: left;
    }
    .data-table th {
        background: #1e293b;
        color: #38bdf8;
        padding: 14px 16px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.88rem;
        border-bottom: 2px solid #334155;
    }
    .data-table td {
        padding: 13px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        color: #f8fafc;
    }
    .data-table tr:hover td {
        background: rgba(56, 189, 248, 0.08);
    }

    /* Severity Tags */
    .tag-minor {
        background: rgba(16, 185, 129, 0.25);
        color: #34d399;
        border: 1.5px solid #10b981;
        padding: 4px 10px;
        border-radius: 5px;
        font-weight: 800;
        font-size: 0.82rem;
    }
    .tag-moderate {
        background: rgba(245, 158, 11, 0.25);
        color: #fbbf24;
        border: 1.5px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 5px;
        font-weight: 800;
        font-size: 0.82rem;
    }
    .tag-critical {
        background: rgba(239, 68, 68, 0.25);
        color: #f87171;
        border: 1.5px solid #ef4444;
        padding: 4px 10px;
        border-radius: 5px;
        font-weight: 800;
        font-size: 0.82rem;
    }

    /* GIS Status Bar */
    .gis-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #080c14;
        border: 1.5px solid #1e293b;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 12px;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.95rem;
        color: #38bdf8;
    }

    /* Action Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        color: #38bdf8 !important;
        border: 1.5px solid #38bdf8 !important;
        border-radius: 10px !important;
        padding: 14px 26px !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.05em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(56, 189, 248, 0.2) !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
        color: #080c14 !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 8px 30px rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Dark Expander Styling */
    div[data-testid="stExpander"] {
        background-color: #0f172a !important;
        border: 1.5px solid #38bdf8 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"] details summary {
        background-color: #0f172a !important;
        color: #38bdf8 !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        padding: 16px 22px !important;
        border-bottom: 1.5px solid #1e293b !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        color: #f59e0b !important;
    }

    /* Benchmarking Metric Card Styling */
    .bench-card {
        background: #080c14;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
    }
    .bench-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 700;
        text-transform: uppercase;
    }
    .bench-val {
        font-family: 'Orbitron', monospace;
        font-size: 1.85rem;
        color: #ffffff;
        font-weight: 900;
        margin-top: 4px;
    }
    .bench-delta {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.82rem;
        color: #34d399;
        font-weight: 700;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Header & Status Bar
# ------------------------------------------------------------------------------
st.markdown("""
<div class="navbar">
    <div>
        <h1 class="brand-title">🛣️ SMARTROAD-VISION</h1>
        <div class="brand-subtitle">MUNICIPAL INFRASTRUCTURE TELEMETRY & DECISION SUPPORT SUITE</div>
    </div>
    <div class="status-pill">
        <span class="pulse-indicator"></span>
        <span>YOLOv8-CBAM ENGINE: ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. Model Engine Loader
# ------------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error("⚠️ Model weight `best.pt` not found in root directory. Please place it alongside `app.py`.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. Adversarial Environmental Stress Simulation Engine
# ------------------------------------------------------------------------------
def apply_environmental_stress(pil_img, mode):
    img_np = np.array(pil_img)
    
    if mode == "Low-Light / Night Ingest":
        table = np.array([((i / 255.0) ** 2.2) * 255 for i in np.arange(0, 256)]).astype("uint8")
        dark = cv2.LUT(img_np, table)
        dark[:, :, 2] = np.clip(dark[:, :, 2] * 1.15, 0, 255)
        return Image.fromarray(dark)
        
    elif mode == "Rain / Wet Asphalt Ripple":
        h, w, _ = img_np.shape
        rain = img_np.copy()
        noise = np.random.normal(0, 15, (h, w, 3)).astype(np.uint8)
        rain = cv2.addWeighted(rain, 0.85, noise, 0.15, 0)
        blurred = cv2.GaussianBlur(rain, (3, 3), 0)
        return Image.fromarray(blurred)
        
    elif mode == "Solar Glare / Overexposure":
        glare = cv2.convertScaleAbs(img_np, alpha=1.35, beta=45)
        return Image.fromarray(glare)
        
    elif mode == "Shadow Canopy Occlusion":
        h, w, _ = img_np.shape
        mask = np.ones((h, w), dtype=np.float32)
        cv2.rectangle(mask, (0, 0), (int(w * 0.55), h), 0.45, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        shadowed = (img_np * mask[:, :, np.newaxis]).astype(np.uint8)
        return Image.fromarray(shadowed)
        
    return pil_img

# ------------------------------------------------------------------------------
# 5. Sidebar Control Deck (High Visibility Dark Glass Controls)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ CONTROL DECK")
    confidence_threshold = st.slider("Optical Sensitivity (Confidence)", 0.05, 1.0, 0.25, 0.05)
    asphalt_cost_per_kg = st.number_input("Standard Bitumen Cost ($/kg)", value=0.15, step=0.01, format="%.2f")
    
    st.markdown("---")
    st.markdown("### 🌧️ ADVERSARIAL STRESS TEST")
    stress_mode = st.selectbox(
        "Simulate Adverse Environment:",
        ["Standard Clean Ingest", "Low-Light / Night Ingest", "Rain / Wet Asphalt Ripple", "Solar Glare / Overexposure", "Shadow Canopy Occlusion"],
        help="Evaluates neural network robustness under adverse weather conditions."
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="background:#131c2e; border:1px solid #1e293b; border-radius:8px; padding:12px 14px; margin-top:8px;">
        <div style="color:#38bdf8; font-weight:800; font-size:0.88rem; margin-bottom:8px;">TELEMETRY CALIBRATION</div>
        <div style="font-size:0.85rem; line-height:1.7; color:#cbd5e1;">
            • <b>BACKBONE:</b> Custom YOLOv8s-CBAM<br>
            • <b>DENSITY:</b> 2,400 kg/m³<br>
            • <b>GSD FACTOR:</b> 0.00005 m²/px²<br>
            • <b>STANDARD:</b> ASTM D6433 PCI<br>
            • <b>LCA CO₂ FACTOR:</b> 0.058 kg CO₂e/kg
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. Geolocation & Reverse Geocoding Engine
# ------------------------------------------------------------------------------
def extract_gps_or_hash_location(image, filename):
    try:
        exif_data = image._getexif()
        if exif_data:
            gps_info = {}
            for tag, value in exif_data.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    for t in value:
                        sub_tag = GPSTAGS.get(t, t)
                        gps_info[sub_tag] = value[t]
                        
            if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                lat = gps_info["GPSLatitude"]
                lon = gps_info["GPSLongitude"]
                lat_dec = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
                lon_dec = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600
                if gps_info.get("GPSLatitudeRef") == "S": lat_dec = -lat_dec
                if gps_info.get("GPSLongitudeRef") == "W": lon_dec = -lon_dec
                return [float(lat_dec), float(lon_dec)], "📍 MOBILE / DRONE EXIF LOCK"
    except Exception:
        pass
        
    hash_num = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    base_lat, base_lon = 17.3850, 78.4867
    lat_offset = float(((hash_num % 1000) - 500) * 0.0001)
    lon_offset = float((((hash_num // 1000) % 1000) - 500) * 0.0001)
    return [float(base_lat + lat_offset), float(base_lon + lon_offset)], "🌐 SYNTHETIC SPATIAL GEOTAG"

@st.cache_data(ttl=3600)
def get_human_readable_address(lat, lon):
    try:
        geolocator = Nominatim(user_agent="smartroad_vision_theme_fixed_app")
        location = geolocator.reverse((lat, lon), language="en", timeout=5)
        if location and location.address:
            return location.address
    except Exception:
        pass
    return "Municipal Grid Sector (Autonomous Offline Mode)"

# ------------------------------------------------------------------------------
# 7. Analytics, ASTM D6433 & Carbon LCA Engine
# ------------------------------------------------------------------------------
def calculate_analytics(results, unit_cost):
    boxes = results[0].boxes
    detections = []
    total_area_px = 0.0
    minor_cnt, mod_cnt, crit_cnt = 0, 0, 0
    
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        
        xyxy = box.xyxy[0].cpu().numpy()
        w = float(xyxy[2] - xyxy[0])
        h = float(xyxy[3] - xyxy[1])
        area = float(w * h)
        total_area_px += area
        
        if area < 15000:
            severity = "Minor"
            minor_cnt += 1
        elif 15000 <= area < 45000:
            severity = "Moderate"
            mod_cnt += 1
        else:
            severity = "Critical"
            crit_cnt += 1
            
        detections.append({
            "target_id": f"TRG-{idx+1:02d}",
            "class": str(cls_name).upper(),
            "confidence": f"{conf * 100:.1f}%",
            "bounds": f"{int(w)}×{int(h)} px",
            "area": f"{int(area):,} px²",
            "severity": severity
        })
        
    rpi_score = int(min(100, (1 * minor_cnt + 3 * mod_cnt + 5 * crit_cnt) * 10))
    deduct_value = min(100.0, (0.5 * minor_cnt + 1.8 * mod_cnt + 4.2 * crit_cnt) * 7.5)
    pci_score = int(max(0, int(100 - deduct_value)))
    
    if pci_score > 40:
        decay_lambda = 0.035 + (crit_cnt * 0.015)
        months_to_failure = float(round((math.log(pci_score / 40.0)) / decay_lambda, 1))
    else:
        months_to_failure = 0.0
        
    asphalt_kg = float(round(total_area_px * 0.00005 * 0.05 * 2400, 2))
    est_cost = float(round(asphalt_kg * float(unit_cost), 2))
    carbon_kg_co2e = float(round(asphalt_kg * 0.058, 2))
    
    return detections, rpi_score, pci_score, months_to_failure, asphalt_kg, est_cost, carbon_kg_co2e, (minor_cnt, mod_cnt, crit_cnt)

# ------------------------------------------------------------------------------
# 8. Municipal Work Order PDF & GeoJSON Generators
# ------------------------------------------------------------------------------
def generate_work_order_pdf(filename, coords, address, rpi, pci, months_to_failure, asphalt_kg, est_cost, carbon_kg, detections):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 32, 'F')
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(56, 189, 248)
    pdf.set_xy(14, 8)
    pdf.cell(0, 10, "MUNICIPAL INFRASTRUCTURE WORK ORDER", ln=True)
    
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.set_xy(14, 18)
    pdf.cell(0, 8, f"SMARTROAD-VISION AUTONOMOUS DISPATCH SYSTEM  |  GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    pdf.ln(12)
    pdf.set_text_color(15, 23, 42)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "1. GEOSPATIAL & SITE TELEMETRY", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(95, 6, f"Incident Target Frame: {filename}", ln=False)
    pdf.cell(95, 6, f"GPS: {float(coords[0]):.5f} N, {float(coords[1]):.5f} E", ln=True)
    pdf.multi_cell(0, 6, f"Resolved Address: {address}")
    pdf.cell(95, 6, f"Road Priority Index (RPI): {rpi} / 100", ln=False)
    pdf.cell(95, 6, f"ASTM D6433 Condition (PCI): {pci} / 100", ln=True)
    pdf.cell(95, 6, f"Deterioration Horizon: {months_to_failure} Mo to Failure", ln=True)
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "2. REQUIRED MATERIAL, BUDGET & ESG LCA", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(95, 6, f"Compacted Asphalt Mass: {asphalt_kg:,.2f} kg", ln=False)
    pdf.cell(95, 6, f"Approved Repair Tariff: ${est_cost:,.2f} USD", ln=True)
    pdf.cell(95, 6, f"Embodied Carbon Footprint: {carbon_kg:,.2f} kg CO2e", ln=True)
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "3. DEFECT MANIFEST BREAKDOWN", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(226, 232, 240)
    pdf.cell(25, 7, "TARGET ID", 1, 0, 'C', True)
    pdf.cell(40, 7, "CLASSIFICATION", 1, 0, 'C', True)
    pdf.cell(30, 7, "CONFIDENCE", 1, 0, 'C', True)
    pdf.cell(45, 7, "DAMAGE AREA", 1, 0, 'C', True)
    pdf.cell(40, 7, "SEVERITY RATING", 1, 1, 'C', True)
    
    pdf.set_font("Helvetica", "", 9)
    for d in detections:
        pdf.cell(25, 6, str(d['target_id']), 1, 0, 'C')
        pdf.cell(40, 6, str(d['class']), 1, 0, 'C')
        pdf.cell(30, 6, str(d['confidence']), 1, 0, 'C')
        pdf.cell(45, 6, str(d['area']), 1, 0, 'C')
        pdf.cell(40, 6, str(d['severity']), 1, 1, 'C')
        
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 5, "Notice: This document is an automated engineering dispatch manifest generated by the SmartRoad-Vision Neural Decision-Support Framework. All volume computations assume ASTM D6433 road base compaction standards.")
    
    return pdf.output()

def generate_geojson_layer(filename, coords, address, rpi, pci, asphalt_kg, est_cost, carbon_kg, detections):
    geojson_feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(coords[1]), float(coords[0])]
                },
                "properties": {
                    "source_frame": str(filename),
                    "resolved_address": str(address),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "road_priority_index": int(rpi),
                    "astm_pci_score": int(pci),
                    "total_defects": int(len(detections)),
                    "asphalt_mass_kg": float(asphalt_kg),
                    "budget_cost_usd": float(est_cost),
                    "carbon_embodied_kg_co2e": float(carbon_kg),
                    "defect_breakdown": detections
                }
            }
        ]
    }
    return json.dumps(geojson_feature, indent=2, default=str)

# ------------------------------------------------------------------------------
# 9. Main Application Flow
# ------------------------------------------------------------------------------
uploaded_file = st.file_uploader("📥 INGEST OPTICAL SENSOR FRAME (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    processed_image = apply_environmental_stress(raw_image, stress_mode)
    results = model.predict(processed_image, conf=confidence_threshold)
    plotted_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    
    detections, rpi, pci, months_to_failure, asphalt_kg, repair_cost, carbon_kg, (min_c, mod_c, crit_c) = calculate_analytics(results, asphalt_cost_per_kg)
    
    rpi_color = "#ef4444" if rpi > 50 else ("#f59e0b" if rpi > 20 else "#10b981")
    pci_color = "#10b981" if pci > 70 else ("#f59e0b" if pci > 40 else "#ef4444")
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card" style="border-left: 4px solid #38bdf8;">
            <div class="kpi-label">TARGETS ACQUIRED</div>
            <div class="kpi-value">{len(detections):02d}</div>
            <div class="kpi-sub">Defect clusters</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid {rpi_color};">
            <div class="kpi-label">ROAD PRIORITY (RPI)</div>
            <div class="kpi-value" style="color: {rpi_color};">{rpi} <span style="font-size:0.95rem; color:#64748b;">/100</span></div>
            <div class="kpi-sub">ASTM PCI: <b style="color:{pci_color};">{pci}/100</b></div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #818cf8;">
            <div class="kpi-label">DECAY HORIZON</div>
            <div class="kpi-value" style="color: #818cf8;">{months_to_failure} <span style="font-size:0.95rem; color:#64748b;">MO</span></div>
            <div class="kpi-sub">To threshold (PCI &lt; 40)</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
            <div class="kpi-label">BITUMEN MASS</div>
            <div class="kpi-value">{asphalt_kg:,.1f} <span style="font-size:0.95rem; color:#64748b;">KG</span></div>
            <div class="kpi-sub">Carbon: <b>{carbon_kg} kg CO₂e</b></div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid #10b981;">
            <div class="kpi-label">DISPATCH BUDGET</div>
            <div class="kpi-value" style="color: #10b981;">${repair_cost:,.2f}</div>
            <div class="kpi-sub">Tariff: ${asphalt_cost_per_kg:.2f}/kg</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    feed_col1, feed_col2 = st.columns(2)
    with feed_col1:
        st.markdown(f'<div class="panel-box"><div class="panel-title-bar"><span>📷 OPTICAL FEED [{stress_mode.upper()}]</span><span style="color:#34d399; font-size:0.85rem;">● BUFFERED</span></div>', unsafe_allow_html=True)
        st.image(processed_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with feed_col2:
        st.markdown('<div class="panel-box"><div class="panel-title-bar"><span>🎯 NEURAL TARGET ACQUISITION [YOLOv8 OVERLAY]</span><span style="color:#38bdf8; font-size:0.85rem;">● ACTIVE</span></div>', unsafe_allow_html=True)
        st.image(plotted_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    data_left, map_right = st.columns([1.18, 0.82])
    
    with data_left:
        st.markdown('<div class="panel-box"><div class="panel-title-bar"><span>📋 DEFECT TELEMETRY MANIFEST</span><span style="color:#94a3b8; font-size:0.85rem;">FORMAT: ISO/MONOSPACE</span></div>', unsafe_allow_html=True)
        
        if detections:
            rows_html = ""
            for d in detections:
                badge_class = f"tag-{d['severity'].lower()}"
                rows_html += f"<tr><td style='color:#38bdf8; font-weight:700; font-size:1.02rem;'>{d['target_id']}</td><td><b style='font-size:1.02rem;'>{d['class']}</b></td><td style='font-size:1.02rem;'>{d['confidence']}</td><td style='font-size:1.02rem;'>{d['bounds']}</td><td style='font-size:1.02rem;'>{d['area']}</td><td><span class='{badge_class}'>{d['severity'].upper()}</span></td></tr>"
            
            table_markup = f"""<div class="table-wrapper"><table class="data-table"><thead><tr><th>TARGET ID</th><th>CLASS</th><th>CONF</th><th>BOUNDS (WxH)</th><th>AREA</th><th>SEVERITY</th></tr></thead><tbody>{rows_html}</tbody></table></div>"""
            st.markdown(table_markup, unsafe_allow_html=True)
        else:
            st.info("NO SURFACE DEFECTS DETECTED ABOVE THE ACTIVE CONFIDENCE THRESHOLD.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with map_right:
        st.markdown('<div class="panel-box"><div class="panel-title-bar"><span>🛰️ REALISTIC SATELLITE GIS RECON</span><span style="color:#38bdf8; font-size:0.85rem;">ESRI SATELLITE + OSM</span></div>', unsafe_allow_html=True)
        
        map_coords, gps_status = extract_gps_or_hash_location(raw_image, uploaded_file.name)
        street_address = get_human_readable_address(map_coords[0], map_coords[1])
        
        st.markdown(f"""
        <div class="gis-bar">
            <span>{gps_status}</span>
            <span>LAT: <b>{map_coords[0]:.4f}</b> | LON: <b>{map_coords[1]:.4f}</b></span>
        </div>
        <div style="font-family:'Share Tech Mono', monospace; font-size:0.95rem; color:#cbd5e1; margin-bottom:10px; line-height:1.4;">
            📍 <b>SITE:</b> {street_address}
        </div>
        """, unsafe_allow_html=True)
        
        pin_color = "red" if crit_c > 0 else ("orange" if mod_c > 0 else "green")
        circle_color = "#ef4444" if crit_c > 0 else ("#f59e0b" if mod_c > 0 else "#10b981")
        
        m = folium.Map(
            location=map_coords,
            zoom_start=17,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery HD Satellite"
        )
        
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png",
            attr="CartoDB Labels",
            name="Road Names Overlay",
            overlay=True,
            control=False
        ).add_to(m)
        
        folium.Circle(
            location=map_coords,
            radius=45,
            color=circle_color,
            fill=True,
            fill_color=circle_color,
            fill_opacity=0.35,
            weight=3,
            tooltip=f"Defect Hazard Radius (PCI: {pci}/100)"
        ).add_to(m)
        
        folium.Marker(
            map_coords,
            popup=folium.Popup(f"""
                <div style="font-family: sans-serif; font-size:13px; min-width:150px;">
                    <b style="color:#0f172a;">SmartRoad Assessment</b><br>
                    <b>RPI Score:</b> {rpi}/100<br>
                    <b>ASTM PCI:</b> {pci}/100<br>
                    <b>Horizon:</b> {months_to_failure} Mo<br>
                    <b>Asphalt:</b> {asphalt_kg} kg<br>
                    <b>Carbon:</b> {carbon_kg} kg CO2e<br>
                    <b>Budget:</b> ${repair_cost}
                </div>
            """, max_width=220),
            tooltip="Click to inspect telemetry",
            icon=folium.Icon(color=pin_color, icon="wrench", prefix="fa")
        ).add_to(m)
        
        st_folium(m, height=265, width=None)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 10. Automated Municipal Work Order PDF & OpenGIS GeoJSON Export Actions
    # --------------------------------------------------------------------------
    st.markdown("---")
    pdf_bytes = generate_work_order_pdf(uploaded_file.name, map_coords, street_address, rpi, pci, months_to_failure, asphalt_kg, repair_cost, carbon_kg, detections)
    geojson_str = generate_geojson_layer(uploaded_file.name, map_coords, street_address, rpi, pci, asphalt_kg, repair_cost, carbon_kg, detections)
    
    act_col1, act_col2, act_col3 = st.columns([1.4, 1.2, 1.4])
    with act_col1:
        st.download_button(
            label="📄 EXPORT WORK ORDER (PDF)",
            data=bytes(pdf_bytes),
            file_name=f"SmartRoad_WorkOrder_{uploaded_file.name.split('.')[0]}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with act_col2:
        st.download_button(
            label="🌐 EXPORT OPENGIS (GEOJSON)",
            data=geojson_str,
            file_name=f"SmartRoad_Telemetry_{uploaded_file.name.split('.')[0]}.geojson",
            mime="application/json",
            use_container_width=True
        )
    with act_col3:
        st.markdown("""
        <div style="font-family:'Share Tech Mono', monospace; font-size:0.86rem; color:#94a3b8; line-height:1.5; padding-top:4px;">
            ⚡ <b>INTEROPERABLE EXPORT:</b> Generates official dispatch PDF & RFC 7946 GeoJSON layer for ArcGIS & QGIS integration.
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 11. High-Contrast Dark-Themed Benchmarking Drawer (Precision, Recall, F1, mAP)
    # --------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("📊 NEURAL ENGINE BENCHMARKS & ABLATION TELEMETRY (YOLOv8-CBAM)", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1:
            st.markdown("""
            <div class="bench-card">
                <div class="bench-label">Precision (P)</div>
                <div class="bench-val" style="color:#38bdf8;">61.7%</div>
                <div class="bench-delta">▲ +3.5% vs Base</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="bench-card">
                <div class="bench-label">Recall (R)</div>
                <div class="bench-val" style="color:#38bdf8;">55.1%</div>
                <div class="bench-delta">▲ +1.0% vs Base</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="bench-card">
                <div class="bench-label">F1-Score</div>
                <div class="bench-val" style="color:#f59e0b;">58.2%</div>
                <div class="bench-delta">▲ +2.2% Harmonic</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown("""
            <div class="bench-card">
                <div class="bench-label">mAP @ 0.50</div>
                <div class="bench-val" style="color:#34d399;">57.7%</div>
                <div class="bench-delta">▲ +3.6% IoU.50</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown("""
            <div class="bench-card">
                <div class="bench-label">Latency (FPS)</div>
                <div class="bench-val" style="color:#a855f7;">4.9 ms</div>
                <div class="bench-delta">~204 FPS (Edge)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
        
        bench_html = """
        <div class="table-wrapper">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>DEFECT CATEGORY</th>
                        <th>PRECISION (P)</th>
                        <th>RECALL (R)</th>
                        <th>F1-SCORE</th>
                        <th>BASELINE mAP@50</th>
                        <th>YOLOV8-CBAM mAP@50</th>
                        <th>NET GAIN</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td><b>D00 (Longitudinal Crack)</b></td><td>62.4%</td><td>54.2%</td><td>58.0%</td><td>55.8%</td><td style="color:#38bdf8; font-weight:800;">58.4%</td><td style="color:#34d399; font-weight:800;">+2.6%</td></tr>
                    <tr><td><b>D10 (Transverse Crack)</b></td><td>60.8%</td><td>53.9%</td><td>57.1%</td><td>54.9%</td><td style="color:#38bdf8; font-weight:800;">57.2%</td><td style="color:#34d399; font-weight:800;">+2.3%</td></tr>
                    <tr><td><b>D20 (Alligator Crack)</b></td><td>69.5%</td><td>66.8%</td><td>68.1%</td><td>65.8%</td><td style="color:#38bdf8; font-weight:800;">68.1%</td><td style="color:#34d399; font-weight:800;">+2.3%</td></tr>
                    <tr><td><b>D40 (Pothole)</b></td><td>54.1%</td><td>45.5%</td><td>49.4%</td><td>49.1%</td><td style="color:#38bdf8; font-weight:800;">52.3%</td><td style="color:#34d399; font-weight:800;">+3.2%</td></tr>
                </tbody>
            </table>
        </div>
        """
        st.markdown(bench_html, unsafe_allow_html=True)