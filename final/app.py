import cv2
import numpy as np
import pandas as pd
import time
import os
import threading
import requests
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
from sort import Sort
from sort import Sort
from werkzeug.utils import secure_filename
from datetime import datetime
import smtplib
from email.message import EmailMessage


# ================= CONFIG =================
app = Flask(__name__)

# Basic Routes data (in-memory for demo, ideally from DB or CSV)
DENSITY_HISTORY_LOG = []

# ================= EMAIL CONFIG =================
SENDER_EMAIL = "vikasyk205@gmail.com"
APP_PASSWORD = "dkfs dvnm rezn sfwx "
RECEIVER_EMAIL = "vikasyk2005@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587



UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

SYSTEM_ID = "BUS_CAM_01"
CLOUD_ALERT_API = "https://cloud-alert-dashboard-1.onrender.com/api/alert"

ROI_AREA_SQ_METERS = 15.0

# ================= GLOBAL STATE =================
cap = None
is_running = False
lock = threading.Lock()
last_email_time = 0


current_metrics = {
    "passenger_count": 0,
    "current_density": 0.0,
    "alert_msg": "System Ready",
    "alert_type": "success",
    "is_monitored": False
}

density_history = deque(maxlen=60)

# ================= LOAD ROUTES =================
df_routes = pd.read_csv("archive/routes.csv")
df_routes["route_no"] = df_routes["route_short_name"].astype(str)
split = df_routes["route_long_name"].str.split("-", expand=True)
df_routes["origin"] = split[0].str.strip()
df_routes["destination"] = split[1].str.strip()

# ================= LOAD MODELS =================
person_model = YOLO("yolov8s.pt")
tracker = Sort()
face_net = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320), 0.6, 0.3, 5000
)

# ================= SEND ALERT TO CLOUD =================
def send_alert_to_cloud(route_no, origin, dest, density, threshold):
    payload = {
        "system_id": SYSTEM_ID,
        "route_no": route_no,
        "origin": origin,
        "destination": dest,
        "density": round(float(density), 2),
        "threshold": threshold,
        "alert_type": "error",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        requests.post(CLOUD_ALERT_API, json=payload, timeout=3)
        print("[CLOUD] Alert sent")
    except Exception as e:
        print("[CLOUD ERROR]", e)

# ================= EMAIL FUNCTION =================
def send_email_alert(event_type, image_path):
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = f"🚨 ALERT: {event_type.upper()} DETECTED"

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    msg.set_content(
        f'''
        ALERT TYPE: {event_type.upper()}
        TIME: {timestamp}

        An event has been detected by the AI Smart Surveillance System.
        Please find the attached screenshot for verification.
        '''
    )

    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
            img_name = os.path.basename(image_path)
        
        msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=img_name)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("[INFO] Email alert sent")
    except Exception as e:
        print("[ERROR] Email failed:", e)

# ================= VIDEO STREAM =================
def generate_frames(route_id, max_density, source_type, filename):
    global cap, is_running, last_email_time


    cap = cv2.VideoCapture(
        0 if source_type == "webcam"
        else os.path.join(UPLOAD_FOLDER, filename)
    )

    while cap.isOpened() and is_running:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        face_net.setInputSize((W, H))

        results = person_model(frame, conf=0.4, verbose=False)
        detections = []

        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, float(box.conf[0])])

        tracks = tracker.update(np.array(detections))
        count = len(tracks)

        for t in tracks:
            x1, y1, x2, y2, tid = map(int, t)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        faces = face_net.detect(frame)[1]
        if faces is not None:
            for f in faces:
                x, y, w, h = map(int, f[:4])
                frame[y:y+h, x:x+w] = cv2.GaussianBlur(
                    frame[y:y+h, x:x+w], (51,51), 30
                )

        density = count / ROI_AREA_SQ_METERS

        route = df_routes[df_routes.route_no == route_id].iloc[0]
        origin, dest = route.origin, route.destination

        if density > max_density:
            send_alert_to_cloud(route_id, origin, dest, density, max_density)
            alert_type = "error"
            alert_msg = "Overcrowding Detected"

            # Check for Email Alert (30s cooldown)
            current_time = time.time()
            if current_time - last_email_time > 30:
                last_email_time = current_time
                # Save temp image
                temp_img_path = os.path.join(UPLOAD_FOLDER, f"alert_{int(current_time)}.jpg")
                cv2.imwrite(temp_img_path, frame)
                # Send email in thread
                threading.Thread(target=send_email_alert, args=("OVERCROWDING", temp_img_path)).start()

        else:
            alert_type = "success"
            alert_msg = "Operating Safely"

        with lock:
            current_metrics.update({
                "passenger_count": count,
                "current_density": round(density, 2),
                "alert_msg": alert_msg,
                "alert_type": alert_type,
                "is_monitored": True
            })
            # Add to history
            record = {
                "time": time.strftime("%H:%M:%S"),
                "density": round(density, 2),
                "threshold": max_density
            }
            density_history.append(record)


        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    cap.release()

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(
        request.args.get("route", "500-D"),
        float(request.args.get("density", 2.0)),
        request.args.get("source_type", "webcam"),
        request.args.get("filename")
    ), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/start", methods=["POST"])
def start():
    global is_running
    is_running = True
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def stop():
    global is_running
    is_running = False
    return jsonify({"status": "stopped"})

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return jsonify({"status": "success", "filename": filename})

@app.route("/api/routes")
def get_routes():
    # Convert routes dataframe to JSON-friendly format
    routes_list = df_routes[["route_no", "origin", "destination"]].to_dict(orient="records")
    return jsonify(routes_list)

@app.route("/api/metrics")
def get_metrics():
    return jsonify(current_metrics)

@app.route("/api/history")
def get_history():
    # Return last 60 points
    return jsonify(list(density_history))

@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")

@app.route("/graph")
def graph_page():
    return render_template("graph.html")


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
