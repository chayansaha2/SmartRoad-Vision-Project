import streamlit as st
from ultralytics import YOLO
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pandas as pd
import folium
from streamlit_folium import st_folium
import hashlib

# ------------------------------------------------------------------------------
# 1. Page Configuration & Full Orbitron + Share Tech Mono Design System
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartRoad-Vision • Pavement Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800;900&family=Share+Tech+Mono&display=swap');

    /* Global Dark Canvas */
    .stApp {
        background-color: #070a12;
        background-image: 
            radial-gradient(circle at 12% 15%, rgba(56, 189, 248, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 88% 85%, rgba(245, 158, 11, 0.04) 0%, transparent 40%),
            linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 28px 28px, 28px 28px;
        font-family: 'Share Tech Mono', monospace;
        color: #e2e8f0;
    }

    /* Top Brand Navigation Header */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-top: 3px solid #38bdf8;
        border-radius: 12px;
        padding: 16px 26px;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6);
    }
    .brand-title {
        font-family: 'Orbitron', monospace !important;
        font-size: 1.55rem;
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
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        margin-top: 4px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 9999px;
    }
    .pulse-indicator {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        animation: pulse-ring 1.8s infinite;
    }
    @keyframes pulse-ring {
        0% { transform: scale(0.95); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.6; }
        100% { transform: scale(0.95); opacity: 1; }
    }

    /* KPI Summary Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 22px;
    }
    .kpi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
    }
    .kpi-label {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
    }
    .kpi-value {
        font-family: 'Orbitron', monospace !important;
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 5px;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 3px;
    }

    /* Content Panels */
    .panel-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 18px;
    }
    .panel-title-bar {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.85rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
        margin-bottom: 14px;
    }

    /* Custom Data Table */
    .table-wrapper {
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #1e293b;
        background: #080c14;
    }
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.78rem;
        text-align: left;
    }
    .data-table th {
        background: #1e293b;
        color: #94a3b8;
        padding: 10px 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.7rem;
        border-bottom: 1px solid #334155;
    }
    .data-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        color: #f1f5f9;
        font-family: 'Share Tech Mono', monospace !important;
    }
    .data-table tr:hover td {
        background: rgba(56, 189, 248, 0.06);
    }

    /* Severity Tags */
    .tag-minor {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.68rem;
        font-family: 'Share Tech Mono', monospace !important;
    }
    .tag-moderate {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.68rem;
        font-family: 'Share Tech Mono', monospace !important;
    }
    .tag-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.68rem;
        font-family: 'Share Tech Mono', monospace !important;
    }

    /* GIS Status Bar */
    .gis-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #080c14;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 7px 12px;
        margin-bottom: 10px;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.72rem;
        color: #38bdf8;
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
        <span>YOLOv8 ENGINE: ONLINE</span>
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
# 4. Sidebar Control Deck
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ CONTROL DECK")
    confidence_threshold = st.slider("Optical Sensitivity (Confidence)", 0.05, 1.0, 0.25, 0.05)
    asphalt_cost_per_kg = st.number_input("Standard Bitumen Cost ($/kg)", value=0.15, step=0.01, format="%.2f")
    st.markdown("---")
    st.markdown("""
    **TELEMETRY CALIBRATION:**
    - `BACKBONE:` Custom YOLOv8s Recon
    - `ASPHALT DENSITY:` $2,400\\text{ kg/m}^3$
    - `GSD SCALE FACTOR:` $0.00005\\text{ m}^2/\\text{px}^2$
    """)

# ------------------------------------------------------------------------------
# 5. Geolocation Telemetry Resolver
# ------------------------------------------------------------------------------
def extract_gps_or_hash_location(image, filename):
    """Extracts actual EXIF GPS coordinates or creates deterministic coordinates for sample images."""
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
                return [lat_dec, lon_dec], "📍 MOBILE EXIF SATELLITE LOCK"
    except Exception:
        pass
        
    hash_num = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    base_lat, base_lon = 17.3850, 78.4867
    lat_offset = ((hash_num % 1000) - 500) * 0.0001
    lon_offset = (((hash_num // 1000) % 1000) - 500) * 0.0001
    return [base_lat + lat_offset, base_lon + lon_offset], "🌐 SYNTHETIC SPATIAL GEOTAG"

# ------------------------------------------------------------------------------
# 6. Analytics & Mathematics Engine
# ------------------------------------------------------------------------------
def calculate_analytics(results, unit_cost):
    boxes = results[0].boxes
    detections = []
    total_area_px = 0
    minor_cnt, mod_cnt, crit_cnt = 0, 0, 0
    
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        
        xyxy = box.xyxy[0].cpu().numpy()
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        area = w * h
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
            "class": cls_name.upper(),
            "confidence": f"{conf * 100:.1f}%",
            "bounds": f"{int(w)}×{int(h)} px",
            "area": f"{int(area):,} px²",
            "severity": severity
        })
        
    rpi_score = min(100, (1 * minor_cnt + 3 * mod_cnt + 5 * crit_cnt) * 10)
    asphalt_kg = round(total_area_px * 0.00005 * 0.05 * 2400, 2)
    est_cost = round(asphalt_kg * unit_cost, 2)
    
    return detections, rpi_score, asphalt_kg, est_cost, (minor_cnt, mod_cnt, crit_cnt)

# ------------------------------------------------------------------------------
# 7. Main Application Flow
# ------------------------------------------------------------------------------
uploaded_file = st.file_uploader("📥 INGEST OPTICAL SENSOR FRAME (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    
    # Run YOLO Model Inference
    results = model.predict(raw_image, conf=confidence_threshold)
    plotted_img = results[0].plot()
    
    # Compute Analytics
    detections, rpi, asphalt_kg, repair_cost, (min_c, mod_c, crit_c) = calculate_analytics(results, asphalt_cost_per_kg)
    
    # Severity Color Rules
    rpi_color = "#ef4444" if rpi > 50 else ("#f59e0b" if rpi > 20 else "#10b981")
    rpi_status = "CRITICAL RISK" if rpi > 50 else ("ELEVATED RISK" if rpi > 20 else "SURFACE STABLE")
    
    # Summary KPI Cards
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card" style="border-left: 3px solid #38bdf8;">
            <div class="kpi-label">TARGETS ACQUIRED</div>
            <div class="kpi-value">{len(detections):02d}</div>
            <div class="kpi-sub">Total surface defect clusters</div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid {rpi_color};">
            <div class="kpi-label">ROAD PRIORITY INDEX (RPI)</div>
            <div class="kpi-value" style="color: {rpi_color};">{rpi} <span style="font-size:0.85rem; color:#64748b;">/100</span></div>
            <div class="kpi-sub">Status: <b>{rpi_status}</b></div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid #f59e0b;">
            <div class="kpi-label">BITUMEN MASS ESTIMATE</div>
            <div class="kpi-value">{asphalt_kg:,.2f} <span style="font-size:0.85rem; color:#64748b;">KG</span></div>
            <div class="kpi-sub">Compaction density: 2.4 t/m³</div>
        </div>
        <div class="kpi-card" style="border-left: 3px solid #10b981;">
            <div class="kpi-label">PROJECTED DISPATCH COST</div>
            <div class="kpi-value" style="color: #10b981;">${repair_cost:,.2f}</div>
            <div class="kpi-sub">At ${asphalt_cost_per_kg:.2f}/kg municipal rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dual Visual Feeds
    feed_col1, feed_col2 = st.columns(2)
    with feed_col1:
        st.markdown('<div class="panel-box"><div class="panel-title-bar"><span>📷 OPTICAL FEED [RAW INGEST]</span><span style="color:#34d399; font-size:0.75rem;">● BUFFERED</span></div>', unsafe_allow_html=True)
        st.image(raw_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with feed_col2:
        st.markdown('<div class="panel-box"><div class="panel-title-bar"><span>🎯 NEURAL TARGET ACQUISITION [YOLOv8 OVERLAY]</span><span style="color:#38bdf8; font-size:0.75rem;">● ACTIVE</span></div>', unsafe_allow_html=True)
        st.image(plotted_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Manifest Table & High-Visibility GIS Map
    data_left, map_right = st.columns([1.15, 0.85])
    
    with data_left:
        st.markdown('<div class="panel-box"><div class="panel-title-bar"><span>📋 DEFECT TELEMETRY MANIFEST</span><span style="color:#94a3b8; font-size:0.75rem;">FORMAT: ISO/MONOSPACE</span></div>', unsafe_allow_html=True)
        
        if detections:
            rows_html = ""
            for d in detections:
                badge_class = f"tag-{d['severity'].lower()}"
                rows_html += f"<tr><td style='color:#38bdf8; font-weight:600;'>{d['target_id']}</td><td><b>{d['class']}</b></td><td>{d['confidence']}</td><td>{d['bounds']}</td><td>{d['area']}</td><td><span class='{badge_class}'>{d['severity'].upper()}</span></td></tr>"
            
            table_markup = f"""<div class="table-wrapper"><table class="data-table"><thead><tr><th>TARGET ID</th><th>CLASS</th><th>CONF</th><th>BOUNDS (WxH)</th><th>AREA</th><th>SEVERITY</th></tr></thead><tbody>{rows_html}</tbody></table></div>"""
            st.markdown(table_markup, unsafe_allow_html=True)
        else:
            st.info("NO SURFACE DEFECTS DETECTED ABOVE THE ACTIVE CONFIDENCE THRESHOLD.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with map_right:
        st.markdown('<div class="panel-box"><div class="panel-title-bar"><span>🗺️ GIS GEOSPATIAL INTELLIGENCE GRID</span><span style="color:#38bdf8; font-size:0.75rem;">EPSG:4326</span></div>', unsafe_allow_html=True)
        
        map_coords, gps_status = extract_gps_or_hash_location(raw_image, uploaded_file.name)
        
        # Telemetry Status Bar
        st.markdown(f"""
        <div class="gis-bar">
            <span>{gps_status}</span>
            <span>LAT: <b>{map_coords[0]:.4f}</b> | LON: <b>{map_coords[1]:.4f}</b></span>
        </div>
        """, unsafe_allow_html=True)
        
        pin_color = "red" if crit_c > 0 else ("orange" if mod_c > 0 else "green")
        circle_color = "#ef4444" if crit_c > 0 else ("#f59e0b" if mod_c > 0 else "#10b981")
        
        # High-Visibility Vector Map
        m = folium.Map(location=map_coords, zoom_start=16, tiles="CartoDB positron")
        
        # Dynamic spatial hazard impact circle
        folium.Circle(
            location=map_coords,
            radius=65,
            color=circle_color,
            fill=True,
            fill_color=circle_color,
            fill_opacity=0.25,
            weight=2,
            tooltip=f"Defect Cluster Hazard Zone (RPI: {rpi}/100)"
        ).add_to(m)
        
        # Central interactive marker
        folium.Marker(
            map_coords,
            popup=folium.Popup(f"""
                <div style="font-family: 'Share Tech Mono', monospace; font-size:12px; min-width:140px;">
                    <b style="color:#0f172a;">SmartRoad Assessment</b><br>
                    <b>RPI Score:</b> {rpi}/100<br>
                    <b>Defects:</b> {len(detections)} Units<br>
                    <b>Asphalt Mass:</b> {asphalt_kg} kg<br>
                    <b>Est. Cost:</b> ${repair_cost}
                </div>
            """, max_width=200),
            tooltip="Click to inspect road health",
            icon=folium.Icon(color=pin_color, icon="wrench", prefix="fa")
        ).add_to(m)
        
        st_folium(m, height=275, width=None)
        st.markdown("</div>", unsafe_allow_html=True)