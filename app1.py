import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort import Sort
import time
import os
import threading
import smtplib
from email.message import EmailMessage
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime

# Initialize Flask App
app = Flask(__name__)

# Upload Config
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------------------
# EMAIL CONFIG (USER PROVIDED)
# -----------------------------------------
SENDER_EMAIL = "vikasyk205@gmail.com"
APP_PASSWORD = "dkfs dvnm rezn sfwx "
RECEIVER_EMAIL = "vikasyk2005@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# LAST EMAIL TIME (Cooldown to prevent spam)
last_email_time = 0
EMAIL_COOLDOWN = 60  # Seconds (e.g., 1 minute)

# -----------------------------------------
# DATABASE CONFIG
# -----------------------------------------
DB_NAME = "alerts.db"
SYSTEM_ID = "BUS_CAM_01"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            system_id TEXT,
            route_no TEXT,
            origin TEXT,
            destination TEXT,
            density REAL,
            threshold REAL,
            alert_type TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_alert_to_db(route_no, origin, dest, density, threshold, alert_type="error"):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            INSERT INTO alerts 
            (system_id, route_no, origin, destination, density, threshold, alert_type, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            SYSTEM_ID,
            str(route_no),
            origin,
            dest,
            float(density),
            float(threshold),
            alert_type,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()
        print(f"[INFO] Alert logged to DB for Route {route_no}")
    except Exception as e:
        print(f"[ERROR] Failed to log alert to DB: {e}")

# -----------------------------------------
# EMAIL FUNCTION
# -----------------------------------------
def send_email_alert(event_type, image_path):
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = f"🚨 ALERT: {event_type.upper()} DETECTED"

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg.set_content(
        f"""
        ALERT TYPE: {event_type.upper()}
        TIME: {timestamp}

        An event has been detected by the AI Smart Surveillance System.
        Please find the attached screenshot for verification.
        """
    )

    # Attach screenshot
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
            img_name = os.path.basename(image_path)

        msg.add_attachment(
            img_data,
            maintype="image",
            subtype="jpeg",
            filename=img_name
        )

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("[INFO] Email alert sent")
    except Exception as e:
        print("[ERROR] Email failed:", e)

# -----------------------------------------
# GLOBAL STATE & CONSTANTS
# -----------------------------------------
from collections import deque
ROI_AREA_SQ_METERS = 15.0
current_metrics = {
    "passenger_count": 0,
    "current_density": 0.0,
    "alert_msg": "",
    "alert_type": "success",  # success, error, warning
    "is_monitored": False
}
density_history = deque(maxlen=60) # Store last 60 seconds (1 min window approx if 1 FPS, or just last 60 points)

lock = threading.Lock()
# ... (rest of simple stuff not shown)



# Video Capture State
cap = None
is_running = False

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------
def load_route_data():
    # Attempting to use absolute, then relative if failed, or kept per user script
    folder_path = r"C:\Users\Yashaswini Kanthraj\Desktop\dtl model\archive"
    file_path = os.path.join(folder_path, "routes.csv")
    
    if not os.path.exists(file_path):
         # Fallback to local if absolute fails (e.g. if moved)
         file_path = "archive/routes.csv"

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
            # Handle cases where split might not produce 2 columns if delimiter missing
            else:
                 df['origin'] = df['route_long_name']
                 df['destination'] = "End Station"
        else:
            df['origin'] = "Unknown Source"
            df['destination'] = "Unknown Dest"
            
        return df[['route_no', 'origin', 'destination']].drop_duplicates()

    except Exception as e:
        print(f"Dataset error: {e}")
        return pd.DataFrame()

df_routes = load_route_data()

# -----------------------------------------
# LOAD AI MODELS
# -----------------------------------------
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

# Initialize Models
try:
    print("Loading AI Models...")
    person_model, face_net = load_models()
    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.2)
    print("Models Loaded Successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    # We don't stop app here, but monitoring won't work well
    person_model, face_net, tracker = None, None, None

# -----------------------------------------
# VIDEO GENERATOR
# -----------------------------------------
def generate_frames(route_id="500-D", max_density=2.0, source_type='upload', source_path=None):
    global cap, is_running, current_metrics, person_model, face_net, tracker, last_email_time

    if person_model is None:
        return

    # Determine Video Source
    if source_type == 'webcam':
        # 0 is usually the default camera. You might need to allow user to pick index if multiple cams.
        video_source = 0 
    elif source_type == 'upload' and source_path:
        video_source = os.path.join(app.config['UPLOAD_FOLDER'], source_path)
    else:
        # Default fallback
        video_source = "crowd_video2.mp4"

    print(f"Starting video feed from source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    total_unique_passengers = set()

    if not cap.isOpened():
        print(f"Error opening {video_source}")
        is_running = False
        return

    while cap.isOpened() and is_running:
        ret, frame = cap.read()
        if not ret:
            # Loop video for continuous demo or stop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        H, W = frame.shape[:2]
        # face_net requires setting input size for each frame size if it changes, 
        # or just once. Doing it here is safe.
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
        
        current_people_in_frame = 0
        
        for track in tracks:
            # Create int coords
            coords = track.astype(int)
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            track_id = coords[4]
            
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
                    # Apply blur
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (51, 51), 30)

        # --- 4. Logic & Metrics ---
        current_density = current_people_in_frame / ROI_AREA_SQ_METERS
        
        # Get Route Details
        try:
            route_info = df_routes[df_routes['route_no'] == str(route_id)].iloc[0]
            origin = route_info['origin']
            dest = route_info['destination']
            route_desc = f"{origin} -> {dest}"
        except:
            route_desc = "Unknown Route"

        # Alert Logic
        alert_type = "success"
        current_time = time.time()
        
        if current_density > max_density:
            alert_msg = f"HIGH ALERT: Route {route_id} ({route_desc}) Overcrowded! ({current_density:.2f} > {max_density})"
            alert_type = "error" # Maps to danger class in frontend
            
            # --- EMAIL TRIGGER ---
            if (current_time - last_email_time) > EMAIL_COOLDOWN:
                last_email_time = current_time
                
                # Save Screenshot
                screenshot_filename = f"alert_{int(current_time)}.jpg"
                cv2.imwrite(screenshot_filename, frame)
                
                # Send Email in background thread to not block video
                email_thread = threading.Thread(
                    target=send_email_alert, 
                    args=(f"Overcrowding Route {route_id}", screenshot_filename)
                )
                email_thread.start()

                # --- LOG TO DB ---
                log_alert_to_db(route_id, origin, dest, current_density, max_density)

        else:
            alert_msg = f"Route {route_id} ({route_desc}) Operating Safely ({current_density:.2f} P/m²)"

        # Update global metrics safely
        with lock:
            current_metrics["passenger_count"] = len(total_unique_passengers)
            current_metrics["current_density"] = round(current_density, 2)
            current_metrics["alert_msg"] = alert_msg
            current_metrics["alert_type"] = alert_type
            current_metrics["is_monitored"] = True
            
            # Add to history
            timestamp_str = time.strftime("%H:%M:%S")
            density_history.append({
                "time": timestamp_str,
                "density": round(current_density, 2),
                "threshold": max_density
            })

        # Encode Frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if cap:
        cap.release()

    if cap:
        cap.release()

# -----------------------------------------
# FLASK ROUTES
# -----------------------------------------
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        return jsonify({"filename": filename, "status": "success"})
    return jsonify({"error": "Upload failed"}), 500
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def graph_page():
    return render_template('graph.html')

@app.route('/video_feed')
def video_feed():
    # Get params
    route = request.args.get('route', '500-D')
    dim = float(request.args.get('density', 2.0))
    source_type = request.args.get('source_type', 'default') # 'upload', 'webcam', 'default'
    filename = request.args.get('filename', None)
    
    return Response(generate_frames(route, dim, source_type, filename), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    global is_running
    if not is_running:
        is_running = True
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    global is_running
    is_running = False
    with lock:
        current_metrics["is_monitored"] = False
    return jsonify({"status": "stopped"})

@app.route('/api/metrics')
def get_metrics():
    with lock:
        return jsonify(current_metrics)

@app.route('/api/history')
def get_history():
    with lock:
        # Return list of history
        return jsonify(list(density_history))

@app.route('/api/routes')
def get_routes():
    if df_routes.empty:
        return jsonify([])
    
    # Convert dataframe to list of dicts
    routes = []
    for _, row in df_routes.iterrows():
        routes.append({
            "route_no": row['route_no'],
            "origin": row['origin'],
            "destination": row['destination']
        })
    return jsonify(routes)

@app.route('/alerts')
def alerts_dashboard():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
    alerts = c.fetchall()
    conn.close()
    return render_template("alerts.html", alerts=alerts)

if __name__ == '__main__':
    # Debug=True helps with auto-reload during dev
    app.run(host='0.0.0.0', port=5000, debug=True)