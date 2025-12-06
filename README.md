# 🚨 Ethical AI Crowd Monitoring System

### Real-Time Person Detection • Face Blur Privacy Protection • Crowd Density Alerts • YOLO + YuNet + SORT

---

## 📝 **Project Overview**

Public places like metro stations, bus stops, shopping streets, and events often use surveillance cameras for **crowd monitoring**, but storing real video threatens citizen privacy.

This project solves that problem by building an **Ethical AI Crowd Monitoring System** that detects people and crowd density **without storing any video frames** and **automatically blurs all faces** to ensure privacy.

---

## 🎯 **Key Features**

### ✔ **1. Real-Time Person Detection (YOLOv8)**

Fast and accurate detection of people in crowd environments.

### ✔ **2. Automatic Face Blur (YuNet ONNX Model)**

Every detected face is blurred instantly for privacy protection.

### ✔ **3. Crowd Tracking (SORT Algorithm)**

Improves accuracy by tracking people across frames.

### ✔ **4. Crowd Density Calculation**

Estimates the percentage of the frame occupied by people.

### ✔ **5. Real-Time Alert System**

Triggers alerts when a crowd exceeds a threshold (40% by default).
Alerts are logged in **alerts.txt**.

### ✔ **6. Ethical Design — No Video Storage**

Frames are processed in RAM only and instantly deleted.
**No images or videos are saved**, ensuring user privacy.

---

## 🧠 **Tech Stack**

| Component            | Technology                      |
| -------------------- | ------------------------------- |
| Person Detection     | YOLOv8 (yolov8s.pt)             |
| Face Detection       | YuNet Face Detector (ONNX)      |
| Tracking             | SORT Algorithm                  |
| Programming Language | Python                          |
| Libraries            | OpenCV, Ultralytics YOLO, NumPy |
| Input Source         | Webcam / Video File             |

---

## 📂 **Project Structure**

```
📁 Ethical-Crowd-Monitoring
│
├── final.py                          # Main program
├── sort.py                           # SORT tracking algorithm
├── face_detection_yunet_2023mar.onnx # YuNet face detector
├── yolov8s.pt                        # YOLO model (person detection)
├── crowd_video3.mp4                  # Sample input video
├── alerts.txt                        # Generated alert logs
│
└── README.md                         # Project documentation
```

---

## ▶️ **How to Run the Project**

### **1. Install Dependencies**

```bash
pip install ultralytics opencv-python numpy
```

### **2. Download Required Models**

Place these files in the project directory:

* `yolov8s.pt`
* `face_detection_yunet_2023mar.onnx`

### **3. Run the Program**

```bash
python final.py
```

### **4. Quit**

Press **Q** to exit.

---

## 🚧 **How It Works Internally**

### **Step 1 — YOLO Person Detection**

Detects bounding boxes around people.

### **Step 2 — YuNet Face Detection + Blur**

All faces inside the frame are strongly blurred.

### **Step 3 — SORT Tracking**

Tracks each person across frames for stable counting.

### **Step 4 — Density Measurement**

Crowd density = total area of person bounding boxes ÷ frame area.

### **Step 5 — Alert System**

If density > 40%, the system writes an alert to `alerts.txt`.

---

## 🔐 **Ethical AI & Privacy Protection**

This project follows strong ethical principles:

### 🔒 **No video storage**

Frames exist only in RAM.

### 🔍 **Face blur before any processing**

Faces are blurred immediately after detection.

### 🚫 **No identification / face recognition**

System counts people, not identities.

### ✔ **Safe for deployment in public spaces**

---

## 🎥 **Demo Video**

(Add your demonstration link here)
Example:
📌 *Coming soon or upload your video to YouTube*

---

## 🏆 **Why This Project Stands Out**

✔ Solves a real-world problem
✔ Uses ethical AI principles
✔ High accuracy (YOLO + YuNet)
✔ Tracks people like professional CCTV systems
✔ Impressive for hackathons and college submissions

---

## 🤝 Contributing

Pull requests are welcome!
If you find bugs or want to improve the system, feel free to open an issue.

---

## 📜 License

This project is licensed under the MIT License.

---
