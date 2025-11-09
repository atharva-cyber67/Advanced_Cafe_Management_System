import cv2
import time
from yolo_detector import YOLODetector
from tracker import Tracker
from face_recognition import FaceRecognition

def main():
    cap = cv2.VideoCapture(0)
    detector = YOLODetector()
    tracker = Tracker()
    face_rec = FaceRecognition("faces.db")  # <-- Initialize once

    prev_time = 0  # For FPS calculation

    while True:
        start_time = time.time()  # Start time for inference
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
                continue  # skip if no valid face embedding
            customer_id = face_rec.compare_embedding(embedding)
            if customer_id:
                status = "Regular"
            else:
                status = "New"
                face_rec.add_embedding(embedding)

            # Draw status on overlay
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"ID:{track['id']} {status}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Calculate inference time and FPS
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # in milliseconds
        fps = 1.0 / (end_time - prev_time) if prev_time != 0 else 0
        prev_time = end_time

        # Display FPS and inference time on frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Inference: {inference_time:.2f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Cafe Monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
