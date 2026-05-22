import atexit
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
from flask import Flask, Response, jsonify, render_template


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model" / "keras_model.h5"
LABELS_PATH = BASE_DIR / "Model" / "labels.txt"

OFFSET = 20
IMG_SIZE = 300


def load_labels(labels_path: Path) -> list[str]:
    labels = []
    with labels_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            _, label = line.split(" ", 1)
            labels.append(label)
    return labels


class SignLanguageCamera:
    def __init__(self) -> None:
        self.detector = None
        self.classifier = None
        self.labels = load_labels(LABELS_PATH)
        self.cap = None
        self.lock = threading.Lock()
        self.active_streams = 0
        self.current_prediction = "Waiting for hand..."
        self.current_confidence = 0.0
        self.last_frame_time = None
        self.model_ready = False

    def ensure_model_loaded(self) -> None:
        with self.lock:
            if self.model_ready:
                return

            self.detector = HandDetector(maxHands=1)
            self.classifier = Classifier(str(MODEL_PATH), str(LABELS_PATH))
            self.model_ready = True

    def start(self) -> None:
        self.ensure_model_loaded()
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def stop(self) -> None:
        with self.lock:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.cap = None

    def add_client(self) -> None:
        with self.lock:
            self.active_streams += 1

    def remove_client(self) -> None:
        with self.lock:
            self.active_streams = max(0, self.active_streams - 1)
            should_stop = self.active_streams == 0

        if should_stop:
            self.stop()

    def _safe_crop(self, img, x, y, w, h):
        y1 = max(0, y - OFFSET)
        y2 = min(img.shape[0], y + h + OFFSET)
        x1 = max(0, x - OFFSET)
        x2 = min(img.shape[1], x + w + OFFSET)
        return img[y1:y2, x1:x2]

    def _classify_hand(self, img, hand):
        x, y, w, h = hand["bbox"]
        img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        img_crop = self._safe_crop(img, x, y, w, h)

        if img_crop.size == 0 or w == 0 or h == 0:
            return None, None, None

        aspect_ratio = h / w

        if aspect_ratio > 1:
            k = IMG_SIZE / h
            w_cal = math.ceil(k * w)
            img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE))
            w_gap = math.ceil((IMG_SIZE - w_cal) / 2)
            img_white[:, w_gap:w_cal + w_gap] = img_resize
            prediction, index = self.classifier.getPrediction(img_white, draw=False)
        else:
            k = IMG_SIZE / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(img_crop, (IMG_SIZE, h_cal))
            h_gap = math.ceil((IMG_SIZE - h_cal) / 2)
            img_white[h_gap:h_cal + h_gap, :] = img_resize
            prediction, index = self.classifier.getPrediction(img_white, draw=False)

        return prediction, index, img_white

    def read_processed_frame(self):
        self.start()

        with self.lock:
            local_cap = self.cap

        if local_cap is None:
            return None

        success, img = local_cap.read()
        if not success or img is None:
            with self.lock:
                self.current_prediction = "Camera unavailable"
                self.current_confidence = 0.0
            return None

        img_output = img.copy()
        hands, img_for_model = self.detector.findHands(img)
        img_output = img_for_model.copy()

        prediction_text = "No hand detected"
        confidence = 0.0

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]
            prediction, index, _ = self._classify_hand(img_for_model, hand)

            if prediction is not None and index is not None and 0 <= index < len(self.labels):
                prediction_text = self.labels[index]
                confidence = float(prediction[index])

                label_box_end_x = x - OFFSET + 420
                label_box_end_y = y - OFFSET + 10
                cv2.rectangle(
                    img_output,
                    (x - OFFSET, y - OFFSET - 70),
                    (label_box_end_x, label_box_end_y),
                    (46, 204, 113),
                    cv2.FILLED,
                )
                cv2.putText(
                    img_output,
                    prediction_text,
                    (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.5,
                    (15, 23, 42),
                    2,
                )
                cv2.rectangle(
                    img_output,
                    (x - OFFSET, y - OFFSET),
                    (x + w + OFFSET, y + h + OFFSET),
                    (46, 204, 113),
                    4,
                )

        cv2.putText(
            img_output,
            f"Prediction: {prediction_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        with self.lock:
            self.current_prediction = prediction_text
            self.current_confidence = confidence
            self.last_frame_time = time.time()

        ret, buffer = cv2.imencode(".jpg", img_output)
        if not ret:
            return None

        return buffer.tobytes()

    def get_prediction_data(self) -> dict:
        with self.lock:
            is_camera_active = self.cap is not None and self.cap.isOpened()
            return {
                "prediction": self.current_prediction,
                "confidence": round(self.current_confidence, 4),
                "camera_active": is_camera_active,
                "last_frame_time": self.last_frame_time,
            }


app = Flask(__name__)
camera = SignLanguageCamera()


@app.route("/")
def index():
    return render_template("index.html")


def generate_frames():
    camera.add_client()
    try:
        while True:
            frame = camera.read_processed_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
    finally:
        camera.remove_client()


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.route("/prediction")
def prediction():
    return jsonify(camera.get_prediction_data())


@atexit.register
def cleanup():
    camera.stop()


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
