from deepface import DeepFace
import numpy as np
import cv2
import sqlite3
import pickle


class FaceRecognition:
    def __init__(self, db_path="faces.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB
            )
        """)
        self.conn.commit()

    def get_face_embedding(self, face_img):
        # Check if the crop is valid (non-zero size)
        if face_img is None or face_img.size == 0:
            return None
        try:
            embedding = DeepFace.represent(face_img, model_name='ArcFace', enforce_detection=False)
            if not embedding or len(embedding) == 0 or "embedding" not in embedding[0]:
                return None
            return np.array(embedding[0]["embedding"])
        except Exception as e:
            print(f"DeepFace error: {e}")
            return None

    def compare_embedding(self, embedding, threshold=0.7):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embedding FROM faces")
        rows = cursor.fetchall()
        for row in rows:
            db_id, db_emb_blob = row
            db_emb = np.frombuffer(db_emb_blob, dtype=np.float32)
            sim = self.cosine_similarity(embedding, db_emb)
            if sim > threshold:
                return db_id
        return None

    def add_embedding(self, embedding):
        cursor = self.conn.cursor()
        emb_blob = embedding.astype(np.float32).tobytes()
        cursor.execute("INSERT INTO faces (embedding) VALUES (?)", (emb_blob,))
        self.conn.commit()
        return cursor.lastrowid

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))