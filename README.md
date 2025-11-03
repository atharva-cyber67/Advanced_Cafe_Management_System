# Advanced_Cafe_Management_System

An AI-powered system designed to automate café operations using **Computer Vision** and **Machine Learning**. This project integrates:
- **YOLOv8n** for real-time object detection,
- **DeepFace (ArcFace embeddings)** for facial recognition,
- **SORT (Simple Online Realtime Tracking)** for object tracking,
- and **SQLite database** for efficient customer record management.

---

## 🚀 Features
- Real-time customer detection and recognition.
- Automatic attendance and activity tracking.
- Customer movement tracking across defined zones.
- Lightweight implementation using Python, OpenCV, and Ultralytics YOLOv8.
- Extendable for queue management, emotion detection, or order analytics.

---

## 🧠 System Architecture
1. Video stream captured via webcam/CCTV.
2. YOLOv8 detects human figures in each frame.
3. SORT tracker assigns IDs and tracks movement.
4. DeepFace extracts facial embeddings using **ArcFace** and compares them with stored vectors in the SQLite database.
5. Data visualization and analytics dashboards (future extension).

---

## 🧩 Tech Stack
- **Python 3.10+**
- **YOLOv8 (Ultralytics)**
- **DeepFace (ArcFace model)**
- **OpenCV**
- **SQLite3**
- **NumPy**
- **Tkinter (for GUI, if applicable)**

---

## 📊 Dataset and Preprocessing
- YOLOv8 pretrained on **COCO dataset** for human detection.
- Facial embeddings generated dynamically using **DeepFace**.
- SQLite database stores customer names, IDs, and embeddings for efficient retrieval.

---

## 🧩 Installation
```bash
git clone https://github.com/<your-username>/Advanced_Cafe_Management_System.git
cd Advanced_Cafe_Management_System
pip install -r requirements.txt
python main.py
