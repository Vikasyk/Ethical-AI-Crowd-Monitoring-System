import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort import Sort
import time
import os

# -----------------------------------------
# CONSTANTS (Physics Calibration)
# -----------------------------------------
# The visible floor area of the bus in the camera view
ROI_AREA_SQ_METERS = 15.0  

# -----------------------------------------
# PAGE SETUP
# -----------------------------------------
st.set_page_config(layout="wide", page_title="BMTC Crowd Safety Monitor")

st.title("🚌 BMTC Ethical Crowd Monitoring System")
st.markdown("### Real-Time Density Analysis ($P/m^2$)")
st.markdown("---")

# -----------------------------------------
# 1. LOAD DATASET (Your Specific Path)
# -----------------------------------------
@st.cache_data
def load_route_data():
    # 🔴 YOUR SPECIFIC FILE PATH
    folder_path = r"C:\Users\YK\Desktop\DTL\archive"
    file_path = os.path.join(folder_path, "routes.csv")
    
    try:
        df = pd.read_csv(file_path)
        
        # Data Cleaning for BMTC Dataset
        if 'route_short_name' in df.columns:
            df['route_no'] = df['route_short_name'].astype(str)
        elif 'route_id' in df.columns:
             df['route_no'] = df['route_id'].astype(str)
             
        if 'route_long_name' in df.columns:
            split_data = df['route_long_name'].str.split('-', n=1, expand=True)
            if split_data.shape[1] >= 2:
                df['origin'] = split_data[0].str.strip()
                df['destination'] = split_data[1].str.strip()
            else:
                df['origin'] = df['route_long_name']
                df['destination'] = "End Station"
        else:
            df['origin'] = "Unknown Source"
            df['destination'] = "Unknown Dest"
            
        return df[['route_no', 'origin', 'destination']].drop_duplicates()

    except FileNotFoundError:
        st.error(f"❌ Could not find file at: {file_path}")
        return pd.DataFrame()

df_routes = load_route_data()

# -----------------------------------------
# 2. LOAD AI MODELS
# -----------------------------------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8s.pt")
    face_net = cv2.FaceDetectorYN.create(
        model="face_detection_yunet_2023mar.onnx",
        config="",
        input_size=(320, 320),
        score_threshold=0.3,
        nms_threshold=0.3,
        top_k=5000
    )
    return yolo_model, face_net

try:
    person_model, face_net = load_models()
    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.2)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# -----------------------------------------
# 3. SIDEBAR CONFIGURATION
# -----------------------------------------
st.sidebar.header("📍 Route Configuration")

if not df_routes.empty:
    route_list = df_routes['route_no'].unique()
    selected_route_id = st.sidebar.selectbox("Select Route ID", route_list)

    route_details = df_routes[df_routes['route_no'] == selected_route_id].iloc[0]
    source_stn = route_details['origin']
    dest_stn = route_details['destination']

    st.sidebar.success(f"**Route:** {selected_route_id}")
    st.sidebar.info(f"**From:** {source_stn}")
    st.sidebar.info(f"**To:** {dest_stn}")
else:
    st.sidebar.warning("Using Manual Mode (Dataset not found)")
    selected_route_id = st.sidebar.text_input("Enter Route ID", "500-D")
    source_stn = "Source"
    dest_stn = "Destination"

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Safety Calibration")
# UPDATED SLIDER: Now uses People per Square Meter
MAX_DENSITY_THRESHOLD = st.sidebar.slider(
    "Max Density (People/m²)", 
    min_value=0.5, 
    max_value=5.0, 
    value=2.0,
    help="International safety standard is often around 2-3 people per sqm."
)

# -----------------------------------------
# 4. MAIN DASHBOARD LAYOUT
# -----------------------------------------
col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.subheader(f"📷 Live Bus Feed: {selected_route_id}")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 Live Metrics")
    
    # KPI Cards
    kpi1, kpi2 = st.columns(2)
    with kpi1:
        st.markdown("**Passenger Count**")
        count_metric = st.empty()
    with kpi2:
        st.markdown("**Density ($P/m^2$)**")
        density_metric = st.empty()
    
    st.divider()
    st.markdown("### 📢 Status Alert")
    alert_placeholder = st.empty()

# -----------------------------------------
# 5. MONITORING LOOP
# -----------------------------------------
start_btn = st.button("🚀 Start Monitoring System")

if start_btn:
    video_source = "crowd_video2.mp4" 
    cap = cv2.VideoCapture(video_source)
    
    # Track unique IDs over time (Total Passengers)
    total_unique_passengers = set()

    if not cap.isOpened():
        st.error(f"Error: Could not open '{video_source}'. Check file location.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video finished.")
                break

            H, W = frame.shape[:2]
            face_net.setInputSize((W, H))

            # --- 1. YOLO Detection ---
            results = person_model(frame, verbose=False, conf=0.4)
            detections = []

            for box in results[0].boxes:
                if int(box.cls[0]) == 0:  # Person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, conf])

            # --- 2. SORT Tracking ---
            tracks = tracker.update(np.array(detections))
            
            # Counter for people CURRENTLY in this frame (Instantaneous Load)
            current_people_in_frame = 0
            
            for track in tracks:
                x1, y1, x2, y2, track_id = track.astype(int)
                
                current_people_in_frame += 1
                total_unique_passengers.add(track_id)
                
                # Visuals
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- 3. Face Blur ---
            faces = face_net.detect(frame)
            if faces[1] is not None:
                for f in faces[1]:
                    x, y, w_f, h_f = map(int, f[:4])
                    x1 = max(0, x); y1 = max(0, y)
                    x2 = min(W, x + w_f); y2 = min(H, y + h_f)
                    if x2 > x1 and y2 > y1:
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (51, 51), 30)

            # --- 4. NEW ENGINEERING CALCULATION ---
            # Density = N / Area
            current_density = current_people_in_frame / ROI_AREA_SQ_METERS
            
            # Display Video (Fixed Warning)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update Metrics
            count_metric.metric("Total Passengers", len(total_unique_passengers))
            density_metric.metric("Current Density", f"{current_density:.2f} P/m²")

            # --- 5. ALERT LOGIC ---
            if current_density > MAX_DENSITY_THRESHOLD:
                msg = f"""
                🚨 **HIGH ALERT** Route **{selected_route_id}** is overcrowded!
                
                **Location:** {source_stn} ➝ {dest_stn}
                **Density:** {current_density:.2f} people/m² (Limit: {MAX_DENSITY_THRESHOLD})
                """
                alert_placeholder.error(msg)
            else:
                alert_placeholder.success(f"✅ Route {selected_route_id} operating safely ({current_density:.2f} P/m²)")

        cap.release()