import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageStat
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import hashlib
import random

# -------------------------------------------------------------
# 1. Page Configuration & Custom CSS (High-End Modern UI)
# -------------------------------------------------------------
st.set_page_config(
    page_title="SmartRoad-Vision AI | Executive Analytics",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Dark Theme CSS Injection
st.markdown("""
<style>
    /* Global background and font styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    /* Header Card */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 25px;
    }
    .main-title {
        color: #38bdf8;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-top: 6px;
    }
    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px;
        border-left: 5px solid #38bdf8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    /* Status Badges */
    .badge-critical {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-moderate {
        background-color: #78350f;
        color: #fde68a;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-minor {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Header Banner UI
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🛣️ SmartRoad-Vision AI</div>
    <div class="main-subtitle">Automated Road Defect Quantification, Infrastructure Priority Indexing & GIS Intelligence</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. Model Loading Engine
# -------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

try:
    model = load_yolo_model()
    st.sidebar.success("⚡ Vision Engine Active: `best.pt` Loaded")
except Exception as e:
    st.sidebar.error("❌ Model File Missing: Please ensure `best.pt` is in the same directory as `app.py`.")

# Sidebar Settings
st.sidebar.markdown("### 🎛️ Control Panel")
confidence_threshold = st.sidebar.slider("AI Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
asphalt_cost_per_kg = st.sidebar.number_input("Asphalt Unit Cost ($/kg)", value=0.15, step=0.01)

# -------------------------------------------------------------
# 4. Helper Functions: Dynamic GPS & Hash Coordinate Mapping
# -------------------------------------------------------------
def extract_gps_or_hash_location(image, filename):
    """Reads EXIF GPS from phone images, or generates deterministic street pins for web images."""
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
                return [lat_dec, lon_dec], "📍 Mobile EXIF Telemetry Active"
    except Exception:
        pass
        
    # Deterministic Hash Offset for web images (Google Images get distinct map pins)
    hash_object = hashlib.md5(filename.encode())
    hash_num = int(hash_object.hexdigest(), 16)
    
    # Base City Center Coordinates (e.g. Hyderabad / City Center)
    base_lat, base_lon = 17.3850, 78.4867
    lat_offset = ((hash_num % 1000) - 500) * 0.0001
    lon_offset = (((hash_num // 1000) % 1000) - 500) * 0.0001
    
    return [base_lat + lat_offset, base_lon + lon_offset], "🌐 Synthetic Spatial Geotag Generated"

# -------------------------------------------------------------
# 5. Core Mathematical & Analytics Engine
# -------------------------------------------------------------
def calculate_analytics(results, unit_cost):
    boxes = results[0].boxes
    detections = []
    total_area_px = 0
    minor_cnt, mod_cnt, crit_cnt = 0, 0, 0
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        
        xyxy = box.xyxy[0].cpu().numpy()
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        area = w * h
        total_area_px += area
        
        # Area-Based Severity Engine
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
            "Defect Class": cls_name.upper(),
            "Confidence": f"{conf * 100:.1f}%",
            "Dimensions (W×H)": f"{int(w)}×{int(h)} px",
            "Surface Area": f"{int(area):,} px²",
            "Severity Level": severity
        })
        
    # Math Formulas
    rpi_score = min(100, (1 * minor_cnt + 3 * mod_cnt + 5 * crit_cnt) * 10)
    asphalt_kg = round(total_area_px * 0.00005 * 0.05 * 2400, 2)
    est_cost = round(asphalt_kg * unit_cost, 2)
    
    return pd.DataFrame(detections), rpi_score, asphalt_kg, est_cost, (minor_cnt, mod_cnt, crit_cnt)

# -------------------------------------------------------------
# 6. Main Interactive Application Logic
# -------------------------------------------------------------
uploaded_file = st.file_uploader("📥 Drag & Drop Road Surface Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    
    # Grid Layout for Visual Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📷 Optical Feed (Input)")
        st.image(raw_image, width="stretch")
        
    with col2:
        st.markdown("##### 🎯 Neural Network Overlay (Inference)")
        results = model.predict(raw_image, conf=confidence_threshold)
        plotted_img = results[0].plot()
        st.image(plotted_img, width="stretch")
        
    st.markdown("---")
    
    # Run Math Analytics
    df_logs, rpi, asphalt_kg, repair_cost, counts = calculate_analytics(results, asphalt_cost_per_kg)
    
    # Metric Dashboard Display
    st.markdown("### 📊 Executive Decision Support Panel")
    
    m1, m2, m3, m4 = st.columns(4)
    
    m1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Detected Defects</div>
        <div class="metric-value">{len(df_logs)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    rpi_color = "#ef4444" if rpi > 50 else ("#f59e0b" if rpi > 20 else "#10b981")
    m2.markdown(f"""
    <div class="metric-card" style="border-left-color: {rpi_color};">
        <div class="metric-label">Road Priority Index (RPI)</div>
        <div class="metric-value" style="color: {rpi_color};">{rpi} <span style="font-size:1rem;">/ 100</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    m3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Est. Material Required</div>
        <div class="metric-value">{asphalt_kg:,.2f} <span style="font-size:1rem;">kg</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    m4.markdown(f"""
    <div class="metric-card" style="border-left-color: #10b981;">
        <div class="metric-label">Est. Repair Budget</div>
        <div class="metric-value" style="color: #10b981;">${repair_cost:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Table & Map Side-by-Side
    left_panel, right_panel = st.columns([1.1, 0.9])
    
    with left_panel:
        st.markdown("##### 📋 Telemetry Defect Breakdown")
        if not df_logs.empty:
            st.dataframe(df_logs, width="stretch", hide_index=True)
        else:
            st.info("No surface defects detected above current confidence threshold.")
            
    with right_panel:
        st.markdown("##### 🗺️ Spatial GIS Intelligence Map")
        map_coords, gps_source_tag = extract_gps_or_hash_location(raw_image, uploaded_file.name)
        
        st.caption(f"Status: {gps_source_tag}")
        
        # Color Logic for Pin
        pin_color = "red" if counts[2] > 0 else ("orange" if counts[1] > 0 else "green")
        
        m = folium.Map(location=map_coords, zoom_start=16, tiles="CartoDB dark_matter")
        folium.Marker(
            map_coords,
            popup=f"RPI Score: {rpi}/100\nAsphalt: {asphalt_kg} kg\nCost: ${repair_cost}",
            tooltip="Click to view segment health",
            icon=folium.Icon(color=pin_color, icon="wrench", prefix="fa")
        ).add_to(m)
        
        st_folium(m, height=310, width=520)