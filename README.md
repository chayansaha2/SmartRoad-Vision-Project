# 🛣️ SmartRoad-Vision
### *Autonomous Pavement Intelligence, ASTM D6433 Condition Indexing & Municipal Decision-Support Suite*

---

## 📌 Overview
**SmartRoad-Vision** is an end-to-end civil infrastructure intelligence and decision-support pipeline designed to automate road surface defect detection and maintenance prioritization. Moving beyond standard 2D bounding-box detection, the framework connects real-time computer vision with physical civil engineering standards, municipal repair budget forecasting, environmental Life Cycle Assessment (LCA), and OpenGIS telemetry.

---

## 🌟 Key Features

* **Attention-Guided Neural Backbone (YOLOv8-CBAM):** Integrates Convolutional Block Attention Modules (CBAM) into the bottleneck to enhance spatial localization of fine longitudinal, transverse, and alligator cracks.
* **ASTM D6433 Pavement Condition Index (PCI) & RPI Engine:** Translates raw defect detections into standardized municipal road health indices (0–100) and predictive deterioration timelines.
* **Volumetric & ESG Carbon LCA Estimation:** Automatically computes physical damage area ($m^2$), bitumen compaction mass ($kg$), repair cost tariffs, and embodied $CO_2e$ carbon emissions.
* **Adversarial Environmental Stress Testing:** Built-in validation module to evaluate detection robustness against synthetic rain, low-light/night conditions, solar glare, and canopy shadows.
* **Realistic Satellite GIS & Reverse Geocoding:** High-definition Esri aerial imagery layered with street vectors, interactive hazard radius zones, and real-time reverse-geocoded street addresses.
* **1-Click Municipal Work Order & GeoJSON Dispatch:** Generates official engineering dispatch PDFs and RFC 7946-compliant `.geojson` vector files for direct integration into ArcGIS, QGIS, and Google Earth.

---

## 🏗️ System Pipeline

[ Optical Ingest / UAV Frame ]
│
▼
[ YOLOv8s-CBAM Attention Inference ]
│
▼
[ Pixel-to-Physical Geometry Engine (m² & kg) ]
│
▼
[ ASTM D6433 PCI & ESG Carbon LCA Scoring ]
│
▼
[ Photorealistic Satellite GIS & Geocoding Layer ]
│
▼
[ Autonomous PDF Work Order & GeoJSON Layer Export ]


---

## 📊 Benchmark & Ablation Results

| Model Architecture | Precision ($P$) | Recall ($R$) | F1-Score | $mAP@50$ | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline YOLOv8s** | 58.2% | 54.1% | 56.0% | 59.5% | **4.4 ms** |
| **Proposed YOLOv8-CBAM (Ours)** | **61.7%** | **55.1%** | **58.2%** | **57.7%** | **4.9 ms** |

---

## 🛠️ Tech Stack & Dependencies

* **Deep Learning:** PyTorch, Ultralytics YOLOv8, OpenCV
* **Dashboard & Visualization:** Streamlit, Folium, Streamlit-Folium
* **Geospatial & Telemetry:** Geopy, Nominatim, PIL EXIF GPS
* **Document Generation:** FPDF2, OpenGIS GeoJSON (RFC 7946)

---
