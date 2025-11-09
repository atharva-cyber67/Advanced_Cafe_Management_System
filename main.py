import cv2
import time
import csv
from datetime import datetime
from yolo_detector import YOLODetector
from tracker import Tracker
from face_recognition import FaceRecognition
import numpy as np
import os

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    cap = cv2.VideoCapture(0)
    detector = YOLODetector()
    tracker = Tracker()
    face_rec = FaceRecognition("faces.db")

    # --- Visitor Tracking and Time Log ---
    active_visitors = {}  # name -> last_seen_time
    session_times = {}    # name -> {'time_in':..., 'time_out':...}
    log_path = "visits.csv"

    # Create CSV if not exists
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time_In", "Time_Out"])

    prev_time = 0

    print("[INFO] Starting camera... Press ESC to exit.")
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame.shape)

        for track in tracks:
            x, y, w, h = map(int, track["bbox"])
            face_img = frame[y:y + h, x:x + w]
            embedding = face_rec.get_face_embedding(face_img)
            if embedding is None:
                continue

            # --- Compare embeddings ---
            best_match = None
            best_score = 0.0
            stored_embeddings = face_rec.get_all_embeddings()
            for stored_name, stored_embedding in stored_embeddings.items():
                score = cosine_similarity(embedding, stored_embedding)
                if score > best_score:
                    best_score = score
                    best_match = stored_name

            # Threshold tuning (0.6–0.8 works well for cosine)
            if best_score > 0.7:
                name = best_match
                status = "Recognized"
            else:
                name = input("Enter your name: ").strip()
                face_rec.add_embedding(embedding, name)
                status = "New Face Added"

            # --- Time logging ---
            current_time = time.time()
            if name not in active_visitors:
                # new entry
                active_visitors[name] = current_time
                session_times[name] = {"time_in": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "time_out": ""}
                print(f"[INFO] {name} entered at {session_times[name]['time_in']}")

            active_visitors[name] = current_time  # update last seen

            # Draw info
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({status})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Detect Time-Out (not seen for 5 sec) ---
        to_remove = []
        for name, last_seen in active_visitors.items():
            if time.time() - last_seen > 5:  # not seen for 5 sec
                session_times[name]["time_out"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, session_times[name]["time_in"], session_times[name]["time_out"]])
                print(f"[INFO] {name} exited at {session_times[name]['time_out']}")
                to_remove.append(name)
        for name in to_remove:
            active_visitors.pop(name)
            session_times.pop(name)

        # FPS display
        end_time = time.time()
        fps = 1.0 / (end_time - prev_time) if prev_time != 0 else 0
        prev_time = end_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Cafe Monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Session ended. Log saved to visits.csv")

if __name__ == "__main__":
    main()
