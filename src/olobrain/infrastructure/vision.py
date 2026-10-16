"""MediaPipe hand tracking adapter with strict camera-frame privacy."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from olobrain.domain.model import GestureKind, HandObservation


class MediaPipeHandTracker:
    """Convert live video to landmarks; pixels are never emitted, stored, or displayed."""

    def __init__(self, model_path: Path, camera_index: int = 0) -> None:
        self._model_path = model_path
        self._camera_index = camera_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(
        self,
        on_observation: Callable[[HandObservation], None],
        on_status: Callable[[str], None],
    ) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(on_observation, on_status),
            name="olobrain-spatial-input",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self, on_observation, on_status) -> None:
        capture = recognizer = None
        try:
            if not self._model_path.exists():
                raise FileNotFoundError(f"gesture model missing: {self._model_path}")
            os.environ.setdefault("MPLCONFIGDIR", str(self._model_path.parents[1] / ".cache" / "matplotlib"))
            import cv2
            import mediapipe as mp

            def handle_result(result, _output_image, timestamp_ms: int) -> None:
                if not self._stop.is_set():
                    on_observation(self._translate(result, timestamp_ms / 1000.0))

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self._model_path)),
                running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.58,
                min_hand_presence_confidence=0.58,
                min_tracking_confidence=0.58,
                result_callback=handle_result,
            )
            recognizer = mp.tasks.vision.HandLandmarker.create_from_options(options)
            capture, active_index, backend_name = self._open_camera(cv2)
            on_status(f"CÁMARA {active_index} ACTIVA · {backend_name} · LANDMARKS ASYNC")

            last_timestamp_ms = 0
            while not self._stop.is_set():
                success, frame = capture.read()
                if not success:
                    time.sleep(0.04)
                    continue
                frame = cv2.flip(frame, 1)
                if frame.shape[1] > 480:
                    frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(last_timestamp_ms + 1, int(time.monotonic() * 1000))
                last_timestamp_ms = timestamp_ms
                recognizer.detect_async(image, timestamp_ms)
                time.sleep(0.008)
        except Exception as exc:  # pragma: no cover - hardware fallback
            on_status(f"MODO MOUSE ACTIVO · ({exc})")
        finally:
            if recognizer is not None:
                recognizer.close()
            if capture is not None:
                capture.release()

    def _open_camera(self, cv2):
        backends = [("DEFAULT", None)]
        if hasattr(cv2, "CAP_DSHOW"):
            backends.insert(0, ("DSHOW", cv2.CAP_DSHOW))
        for index in (self._camera_index, 0, 1):
            for name, backend in backends:
                cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
                if cap is not None and cap.isOpened():
                    return cap, index, name
                if cap is not None:
                    cap.release()
        raise RuntimeError("No camera device available")

    @classmethod
    def _translate(cls, result, captured_at: float) -> HandObservation:
        hands = getattr(result, "hand_landmarks", None) or ()
        if not hands:
            return HandObservation.empty(captured_at)
        primary = hands[0]
        gesture, confidence = cls._classify_gesture(primary)
        palm_x = sum(float(primary[i].x) for i in (0, 5, 9, 13, 17)) / 5.0
        palm_y = sum(float(primary[i].y) for i in (0, 5, 9, 13, 17)) / 5.0
        pointer_x = float(primary[8].x)
        pointer_y = float(primary[8].y)
        if len(hands) >= 2:
            second = hands[1]
            second_x = sum(float(second[i].x) for i in (0, 5, 9, 13, 17)) / 5.0
            second_y = sum(float(second[i].y) for i in (0, 5, 9, 13, 17)) / 5.0
            span = max(0.08, math_hypot(palm_x - second_x, palm_y - second_y) * 2.4)
        else:
            span = max(0.08, cls._distance(primary, 0, 9) * 4.2)
        landmarks = tuple(
            (float(point.x), float(point.y))
            for hand in hands[:2]
            for point in hand[:21]
        )
        return HandObservation(
            gesture=gesture,
            confidence=confidence,
            palm_x=palm_x,
            palm_y=palm_y,
            pointer_x=pointer_x,
            pointer_y=pointer_y,
            span=span,
            hands=min(2, len(hands)),
            landmarks=landmarks,
            captured_at=captured_at,
        )

    @classmethod
    def _classify_gesture(cls, hand) -> tuple[GestureKind, float]:
        palm_scale = max(cls._distance(hand, 0, 9), 1e-4)
        if cls._distance(hand, 4, 8) < palm_scale * 0.32:
            return GestureKind.PINCH, 0.96

        index = cls._finger_extended(hand, 8, 6)
        middle = cls._finger_extended(hand, 12, 10)
        ring = cls._finger_extended(hand, 16, 14)
        pinky = cls._finger_extended(hand, 20, 18)
        thumb_reference = cls._distance(hand, 3, 5)
        thumb_extended = cls._distance(hand, 4, 5) > thumb_reference * 1.25
        extended_count = sum((index, middle, ring, pinky))

        if extended_count == 4:
            return GestureKind.OPEN_PALM, 0.92
        if index and middle and ring and not pinky:
            return GestureKind.THREE_FINGERS, 0.94
        if index and pinky and not middle and not ring:
            return GestureKind.FLOW, 0.93
        if pinky and not index and not middle and not ring:
            return GestureKind.PINKY, 0.93
        if index and middle and not ring and not pinky:
            return GestureKind.VICTORY, 0.92
        if index and not middle and not ring and not pinky:
            return GestureKind.POINT, 0.90
        if extended_count == 0 and thumb_extended:
            if float(hand[4].y) < float(hand[2].y) - 0.035:
                return GestureKind.THUMB_UP, 0.88
            if float(hand[4].y) > float(hand[2].y) + 0.035:
                return GestureKind.THUMB_DOWN, 0.88
        if extended_count == 0:
            return GestureKind.CLOSED_FIST, 0.86
        return GestureKind.NONE, 0.72

    @classmethod
    def _finger_extended(cls, hand, tip: int, joint: int) -> bool:
        return cls._distance(hand, tip, 0) > cls._distance(hand, joint, 0) * 1.12

    @staticmethod
    def _distance(hand, first: int, second: int) -> float:
        return (
            sum(
                (float(getattr(hand[first], axis)) - float(getattr(hand[second], axis))) ** 2
                for axis in ("x", "y", "z")
            )
            ** 0.5
        )


def math_hypot(x: float, y: float) -> float:
    """Keep the hot camera loop independent from array libraries."""
    return (x * x + y * y) ** 0.5
