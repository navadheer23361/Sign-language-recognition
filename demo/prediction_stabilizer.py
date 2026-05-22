from collections import Counter, deque


class PredictionStabilizer:
    def __init__(
        self,
        labels,
        window_size=7,
        min_confidence=0.55,
        min_votes=4,
    ):
        if min_votes > window_size:
            raise ValueError("min_votes cannot be greater than window_size")

        self.labels = list(labels)
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.min_votes = min_votes
        self.history = deque(maxlen=window_size)
        self.last_stable_label = None
        self.last_stable_confidence = 0.0

    def _make_result(self, label, confidence, raw_label, raw_confidence, is_stable):
        return {
            "label": label,
            "confidence": confidence,
            "raw_label": raw_label,
            "raw_confidence": raw_confidence,
            "is_stable": is_stable,
        }

    def reset(self, label="No hand detected"):
        self.history.clear()
        self.last_stable_label = None
        self.last_stable_confidence = 0.0
        return self._make_result(label, 0.0, None, 0.0, False)

    def update(self, prediction, index):
        raw_label = None
        raw_confidence = 0.0

        if prediction is not None and index is not None and 0 <= index < len(self.labels):
            raw_label = self.labels[index]
            raw_confidence = float(prediction[index])

        if raw_label is not None and raw_confidence >= self.min_confidence:
            self.history.append((raw_label, raw_confidence))
        else:
            self.history.append(None)

        valid_items = [item for item in self.history if item is not None]

        if valid_items:
            counts = Counter(label for label, _ in valid_items)
            candidate, votes = counts.most_common(1)[0]

            if votes >= self.min_votes:
                candidate_confidences = [
                    confidence
                    for label, confidence in valid_items
                    if label == candidate
                ]
                self.last_stable_label = candidate
                self.last_stable_confidence = (
                    sum(candidate_confidences) / len(candidate_confidences)
                )
                return self._make_result(
                    candidate,
                    self.last_stable_confidence,
                    raw_label,
                    raw_confidence,
                    True,
                )

        if self.last_stable_label:
            still_visible = any(
                item is not None and item[0] == self.last_stable_label
                for item in valid_items
            )
            if still_visible:
                return self._make_result(
                    self.last_stable_label,
                    self.last_stable_confidence,
                    raw_label,
                    raw_confidence,
                    False,
                )

        return self._make_result(
            "Detecting...",
            0.0,
            raw_label,
            raw_confidence,
            False,
        )
