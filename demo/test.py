import math
import time
from pathlib import Path

import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model" / "keras_model.h5"
LABELS_PATH = BASE_DIR / "Model" / "labels.txt"

OFFSET = 20
IMG_SIZE = 300
LABELS = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]


def safe_crop(img, x, y, w, h, offset):
    y1 = max(0, y - offset)
    y2 = min(img.shape[0], y + h + offset)
    x1 = max(0, x - offset)
    x2 = min(img.shape[1], x + w + offset)
    return img[y1:y2, x1:x2]


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    detector = HandDetector(maxHands=1)
    classifier = Classifier(str(MODEL_PATH), str(LABELS_PATH))
    last_prediction = None

    try:
        while True:
            success, img = cap.read()
            if not success or img is None:
                time.sleep(0.05)
                continue

            img_output = img.copy()
            hands, img_hands = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand["bbox"]

                if w > 0 and h > 0:
                    img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
                    img_crop = safe_crop(img_hands, x, y, w, h, OFFSET)

                    if img_crop.size > 0:
                        aspect_ratio = h / w

                        if aspect_ratio > 1:
                            k = IMG_SIZE / h
                            w_cal = math.ceil(k * w)
                            img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE))
                            w_gap = math.ceil((IMG_SIZE - w_cal) / 2)
                            img_white[:, w_gap:w_cal + w_gap] = img_resize
                            prediction, index = classifier.getPrediction(img_white, draw=False)
                        else:
                            k = IMG_SIZE / w
                            h_cal = math.ceil(k * h)
                            img_resize = cv2.resize(img_crop, (IMG_SIZE, h_cal))
                            h_gap = math.ceil((IMG_SIZE - h_cal) / 2)
                            img_white[h_gap:h_cal + h_gap, :] = img_resize
                            prediction, index = classifier.getPrediction(img_white, draw=False)

                        if 0 <= index < len(LABELS):
                            current_prediction = LABELS[index]
                            if current_prediction != last_prediction:
                                print(f"Prediction: {current_prediction}")
                                last_prediction = current_prediction

                            cv2.rectangle(
                                img_output,
                                (x - OFFSET, y - OFFSET - 70),
                                (x - OFFSET + 400, y - OFFSET + 10),
                                (0, 255, 0),
                                cv2.FILLED,
                            )
                            cv2.putText(
                                img_output,
                                current_prediction,
                                (x, y - 30),
                                cv2.FONT_HERSHEY_COMPLEX,
                                2,
                                (0, 0, 0),
                                2,
                            )
                            cv2.rectangle(
                                img_output,
                                (x - OFFSET, y - OFFSET),
                                (x + w + OFFSET, y + h + OFFSET),
                                (0, 255, 0),
                                4,
                            )

                        cv2.imshow("ImageCrop", img_crop)
                        cv2.imshow("ImageWhite", img_white)

            cv2.imshow("Image", img_output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
