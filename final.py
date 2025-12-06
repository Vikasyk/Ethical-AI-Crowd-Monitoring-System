import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import time

# -----------------------------------------
# LOAD MODELS
# -----------------------------------------

# YOLO model for person detection
person_model = YOLO("yolov8s.pt")

# YuNet face detector
face_net = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.3,
    nms_threshold=0.3,
    top_k=5000
)

# SORT tracker for accurate counting
tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.2)

# -----------------------------------------
# VIDEO INPUT
# -----------------------------------------

cap = cv2.VideoCapture("crowd_video2.mp4")  # or use 0 for webcam

DENSITY_THRESHOLD = 0.50
ALERT_FILE = "alerts.txt"

total_count = 0
track_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # YuNet input resize
    face_net.setInputSize((W, H))

    # -----------------------------------------
    # 1️⃣ YOLO PERSON DETECTION
    # -----------------------------------------

    results = person_model(frame, verbose=False)
    detections = []  # for SORT
    person_boxes = []  # for density

    for box in results[0].boxes:
        if int(box.cls[0]) == 0:  # person class only
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            detections.append([x1, y1, x2, y2, conf])
            person_boxes.append((x1, y1, x2, y2))

    detections = np.array(detections)

    # -----------------------------------------
    # 2️⃣ SORT TRACKING
    # -----------------------------------------

    tracks = tracker.update(detections)

    # Draw YOLO+SORT tracking boxes
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Count unique IDs
        if track_id not in track_ids:
            track_ids.add(track_id)
            total_count += 1

    # -----------------------------------------
    # 3️⃣ YUNET FACE DETECTION + FACE BLUR
    # -----------------------------------------

    faces = face_net.detect(frame)

    if faces[1] is not None:
        for f in faces[1]:
            x, y, w_face, h_face = map(int, f[:4])

            x1 = max(0, x - int(w_face * 0.25))
            y1 = max(0, y - int(h_face * 0.25))
            x2 = min(W, x + w_face + int(w_face * 0.25))
            y2 = min(H, y + h_face + int(h_face * 0.25))

            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size > 0:
                blur = cv2.GaussianBlur(face_roi, (75, 75), 0)
                frame[y1:y2, x1:x2] = blur

    # -----------------------------------------
    # 4️⃣ CROWD DENSITY CALCULATION
    # -----------------------------------------

    total_area = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in person_boxes)
    density = total_area / (W * H)

    cv2.putText(frame, f"People: {total_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    cv2.putText(frame, f"Density: {density*100:.1f}%", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    # -----------------------------------------
    # 5️⃣ ALERT SYSTEM
    # -----------------------------------------

    if density > DENSITY_THRESHOLD:
        cv2.putText(frame, "⚠ HIGH CROWD DENSITY!", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)

        with open(ALERT_FILE, "a") as f:
            f.write(f"{time.ctime()} | ALERT | Density={density:.3f}, Count={total_count}\n")

    # -----------------------------------------
    # 6️⃣ SHOW OUTPUT
    # -----------------------------------------
    cv2.imshow("YOLO + SORT + YuNet Face Blur + Density Alert", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
