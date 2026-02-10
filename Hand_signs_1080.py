from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================

CAP_W, CAP_H, CAP_FPS = 1920, 1080, 60     # Camera capture
INF_W, INF_H = 640, 360                    # Inference resolution (16:9)
DRAW_DEBUG = True                          # Toggle landmark drawing

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


# ============================================================
# MODEL SETUP
# ============================================================

def ensure_model_file(path: str) -> None:
    if os.path.exists(path):
        return
    print("Downloading MediaPipe hand_landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, path)


# ============================================================
# DRAWING
# ============================================================

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

_cv_line = cv2.line
_cv_circle = cv2.circle


def _to_px(lm, w: int, h: int) -> Tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def draw_hand_landmarks(frame, lm):
    h, w = frame.shape[:2]

    for a, b in HAND_CONNECTIONS:
        ax, ay = _to_px(lm[a], w, h)
        bx, by = _to_px(lm[b], w, h)
        _cv_line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)

    for i, p in enumerate(lm):
        x, y = _to_px(p, w, h)
        r = 6 if i in (4, 8, 12, 16, 20) else 4
        _cv_circle(frame, (x, y), r, (0, 0, 255), -1)


# ============================================================
# GESTURE LOGIC
# ============================================================

@dataclass(frozen=True)
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool


def get_finger_state(lm, handedness: Optional[str]) -> FingerState:
    y_margin = 0.02
    x_margin = 0.01

    index = lm[8].y < lm[6].y - y_margin
    middle = lm[12].y < lm[10].y - y_margin
    ring = lm[16].y < lm[14].y - y_margin
    pinky = lm[20].y < lm[18].y - y_margin

    if handedness == "Right":
        thumb = lm[4].x < lm[3].x - x_margin
    elif handedness == "Left":
        thumb = lm[4].x > lm[3].x + x_margin
    else:
        thumb = abs(lm[4].x - lm[3].x) > 0.04

    return FingerState(thumb, index, middle, ring, pinky)


def _dist2(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def classify_gesture(state: FingerState, lm) -> str:
    if not state.index and not state.middle and not state.ring and not state.pinky:
        return "Thumbs Up" if state.thumb else "Fist"

    if _dist2(lm[4], lm[8]) < 0.055 ** 2 and state.middle and state.ring and state.pinky:
        return "OK"

    if state.middle and not state.index and not state.ring and not state.pinky:
        return "Middle Finger"

    if state.index and state.pinky and not state.middle and not state.ring:
        return "Rock"

    if state.index and state.middle and state.ring and state.pinky:
        return "Open Hand"

    if state.index and state.middle and not state.ring and not state.pinky:
        return "Peace"

    return "Unknown"


# ============================================================
# MAIN
# ============================================================

def main():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(os.cpu_count() or 4)

    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    ensure_model_file(model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,   # ✅ CORRECT PARAMETER
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(
        f"Camera: {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.1f} FPS"
    )

    fps = 0.0
    last_t = time.perf_counter()
    start_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        # --- Inference (downscaled) ---
        inf = cv2.resize(frame, (INF_W, INF_H), interpolation=cv2.INTER_LINEAR)
        inf_rgb = cv2.cvtColor(inf, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, inf_rgb)

        ts_ms = int((time.perf_counter() - start_t) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        gesture = "No hand"
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            if DRAW_DEBUG:
                draw_hand_landmarks(frame, lm)

            handedness = (
                result.handedness[0][0].category_name
                if result.handedness and result.handedness[0]
                else None
            )

            state = get_finger_state(lm, handedness)
            gesture = classify_gesture(state, lm)

        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(frame, gesture, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"{fps:.1f} FPS", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Hand Signs (1080p)", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
