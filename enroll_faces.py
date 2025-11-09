import cv2
from face_recognition import FaceRecognition

def enroll_person():
    name = input("Enter the name of the person: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty!")
        return

    cap = cv2.VideoCapture(0)
    face_rec = FaceRecognition("faces.db")

    captured_faces = []
    print(f"[INFO] Move your head slowly left-right while recording for {name}...")
    print("Press 'c' to capture frame, 'ESC' to finish.")

    while len(captured_faces) < 10:
        ret, frame = cap.read()
        if not ret:
            break

        # Display progress
        cv2.putText(frame, f"Capturing {len(captured_faces)+1}/10", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Enroll Face", frame)

        key = cv2.waitKey(100)
        if key == ord('c'):  # Press 'c' to capture
            h, w, _ = frame.shape
            face_img = frame[h//4:h*3//4, w//4:w*3//4]  # simple center crop
            captured_faces.append(face_img)
            print(f"[INFO] Captured sample {len(captured_faces)}")

        elif key == 27:  # ESC to stop early
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_faces:
        face_rec.add_embeddings_for_person(name, captured_faces, num_samples=len(captured_faces))
        print(f"[INFO] Enrollment complete for {name}!")
    else:
        print("[INFO] No face samples captured, exiting.")

if __name__ == "__main__":
    enroll_person()
