import math
import time
from pathlib import Path

import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

from prediction_stabilizer import PredictionStabilizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model" / "keras_model.h5"
LABELS_PATH = BASE_DIR / "Model" / "labels.txt"

OFFSET = 20
IMG_SIZE = 300
WINDOW_SIZE = 7
MIN_CONFIDENCE = 0.55
MIN_VOTES = 4


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


def safe_crop(img, x, y, w, h, offset):
    y1 = max(0, y - offset)
    y2 = min(img.shape[0], y + h + offset)
    x1 = max(0, x - offset)
    x2 = min(img.shape[1], x + w + offset)
    return img[y1:y2, x1:x2]


def classify_hand(img, hand, classifier):
    x, y, w, h = hand["bbox"]
    img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    img_crop = safe_crop(img, x, y, w, h, OFFSET)

    if img_crop.size == 0 or w == 0 or h == 0:
        return None, None, None, None

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

    return prediction, index, img_crop, img_white


def main():
    labels = load_labels(LABELS_PATH)
    stabilizer = PredictionStabilizer(
        labels,
        window_size=WINDOW_SIZE,
        min_confidence=MIN_CONFIDENCE,
        min_votes=MIN_VOTES,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    detector = HandDetector(maxHands=1)
    classifier = Classifier(str(MODEL_PATH), str(LABELS_PATH))
    last_printed = None

    try:
        while True:
            success, img = cap.read()
            if not success or img is None:
                time.sleep(0.05)
                continue

            img_output = img.copy()
            hands, img_for_model = detector.findHands(img.copy())

            result = stabilizer.reset() if not hands else None
            img_crop = None
            img_white = None

            if hands:
                hand = hands[0]
                x, y, w, h = hand["bbox"]

                prediction, index, img_crop, img_white = classify_hand(
                    img_for_model,
                    hand,
                    classifier,
                )
                result = stabilizer.update(prediction, index)

                cv2.rectangle(
                    img_output,
                    (x - OFFSET, y - OFFSET - 70),
                    (x - OFFSET + 420, y - OFFSET + 10),
                    (0, 255, 0),
                    cv2.FILLED,
                )
                cv2.putText(
                    img_output,
                    result["label"],
                    (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.7,
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

            if result["label"] != last_printed:
                print(
                    f"Stable: {result['label']} | "
                    f"Raw: {result['raw_label'] or 'None'} | "
                    f"Raw confidence: {result['raw_confidence']:.2f}"
                )
                last_printed = result["label"]

            cv2.putText(
                img_output,
                f"Stable: {result['label']}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img_output,
                f"Raw: {result['raw_label'] or 'None'} ({result['raw_confidence'] * 100:.1f}%)",
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 255, 255),
                2,
            )
            cv2.putText(
                img_output,
                f"Threshold: {MIN_CONFIDENCE:.2f} | Window: {WINDOW_SIZE}",
                (20, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 220, 255),
                2,
            )

            cv2.imshow("Stable Image", img_output)

            if img_crop is not None:
                cv2.imshow("Stable ImageCrop", img_crop)
            if img_white is not None:
                cv2.imshow("Stable ImageWhite", img_white)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
