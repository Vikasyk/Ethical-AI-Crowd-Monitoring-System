import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import time

# -----------------------------------------
# CONSTANTS & CALIBRATION
# -----------------------------------------
# 🔴 ESTIMATED PHYSICAL AREA OF THE VIEWABLE FLOOR (in Square Meters)
# You must change this value based on your specific camera view.
# Example: A typical city bus standing area might be around 10 to 15 sq meters.
ROI_AREA_SQ_METERS = 15.0 

# Safety Threshold: 
# 2.0 people/m² = Crowded
# 4.0 people/m² = Dangerous/Jam-packed
DENSITY_LIMIT_PER_SQM = 2.0 

ALERT_FILE = "alerts.txt"

# -----------------------------------------
# LOAD MODELS
# -----------------------------------------
person_model = YOLO("yolov8s.pt")

face_net = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.3,
    nms_threshold=0.3,
    top_k=5000
)

tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.2)

# -----------------------------------------
# VIDEO INPUT
# -----------------------------------------
# cap = cv2.VideoCapture(0)  # Use this for webcam
cap = cv2.VideoCapture("crowd_video2.mp4") # Use this for your downloaded video

total_unique_people = 0
track_ids_history = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    face_net.setInputSize((W, H))

    # -----------------------------------------
    # 1️⃣ YOLO PERSON DETECTION
    # -----------------------------------------
    results = person_model(frame, verbose=False, conf=0.4)
    detections = [] 
    
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:  # person class only
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)

    # -----------------------------------------
    # 2️⃣ SORT TRACKING
    # -----------------------------------------
    tracks = tracker.update(detections)

    current_people_count = 0 # N (Number of people currently in frame)

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        
        # Count people currently in the frame for density calculation
        current_people_count += 1

        # Draw Tracking Boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Count total unique visitors (cumulative)
        if track_id not in track_ids_history:
            track_ids_history.add(track_id)
            total_unique_people += 1

    # -----------------------------------------
    # 3️⃣ YUNET FACE BLUR
    # -----------------------------------------
    faces = face_net.detect(frame)
    if faces[1] is not None:
        for f in faces[1]:
            x, y, w_face, h_face = map(int, f[:4])
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(W, x + w_face); y2 = min(H, y + h_face)
            
            # Basic validation to avoid errors
            if x2 > x1 and y2 > y1:
                face_roi = frame[y1:y2, x1:x2]
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (75, 75), 0)

    # -----------------------------------------
    # 4️⃣ NEW DENSITY CALCULATION (People per sqm)
    # -----------------------------------------
    
    # Formula: Density = N / A(ROI)
    # N = current_people_count
    # A = ROI_AREA_SQ_METERS
    
    density_val = current_people_count / ROI_AREA_SQ_METERS

    # Display Stats
    cv2.putText(frame, f"Current Count (N): {current_people_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.putText(frame, f"Density: {density_val:.2f} p/m2", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    # -----------------------------------------
    # 5️⃣ ALERT SYSTEM
    # -----------------------------------------
    if density_val > DENSITY_LIMIT_PER_SQM:
        cv2.putText(frame, "⚠ HIGH CROWD DENSITY!", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)

        with open(ALERT_FILE, "a") as f:
            f.write(f"{time.ctime()} | ALERT | Density={density_val:.2f} p/sqm, Count={current_people_count}\n")

    # -----------------------------------------
    # 6️⃣ SHOW OUTPUT
    # -----------------------------------------
    cv2.imshow("Ethical AI Crowd Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()