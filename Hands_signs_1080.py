from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================

CAP_W, CAP_H, CAP_FPS = 1920, 1080, 60
INF_W, INF_H = 640, 360
DRAW_DEBUG = True
MAX_HANDS = 2

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


# ============================================================
# MODEL SETUP
# ============================================================

def ensure_model_file(path: str) -> None:
    if not os.path.exists(path):
        print("Downloading MediaPipe hand_landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, path)


# ============================================================
# DRAWING
# ============================================================

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

def _to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def draw_hand_landmarks(frame, lm):
    h, w = frame.shape[:2]

    for a, b in HAND_CONNECTIONS:
        ax, ay = _to_px(lm[a], w, h)
        bx, by = _to_px(lm[b], w, h)
        cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)

    for i, p in enumerate(lm):
        x, y = _to_px(p, w, h)
        r = 6 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, (x, y), r, (0, 0, 255), -1)


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


def _is_thumb_tucked(lm) -> bool:
    """Returns True if thumb is folded against the palm"""
    thumb_tip = lm[4]
    index_mcp = lm[5]
    middle_mcp = lm[9]
    return min(_dist2(thumb_tip, index_mcp), _dist2(thumb_tip, middle_mcp)) < 0.075 ** 2


def classify_gesture(state: FingerState, lm) -> str:
    # Fingers folded
    others_folded = not state.index and not state.middle and not state.ring and not state.pinky

    if others_folded:
        if _is_thumb_tucked(lm):
            return "Fist"
        else:
            return "Thumbs Up"

    # OK sign
    if _dist2(lm[4], lm[8]) < 0.055 ** 2 and state.middle and state.ring and state.pinky:
        return "OK"

    # Middle finger
    if state.middle and not state.index and not state.ring and not state.pinky:
        return "Middle Finger"

    # Rock
    if state.index and state.pinky and not state.middle and not state.ring:
        return "Rock"

    # Open hand
    if state.index and state.middle and state.ring and state.pinky:
        return "Open Hand"

    # Peace
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

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start_t = time.perf_counter()
    last_t = start_t
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        inf = cv2.resize(frame, (INF_W, INF_H))
        inf_rgb = cv2.cvtColor(inf, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, inf_rgb)

        ts_ms = int((time.perf_counter() - start_t) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        if result.hand_landmarks:
            for i, lm in enumerate(result.hand_landmarks):
                draw_hand_landmarks(frame, lm)

                handedness = (
                    result.handedness[i][0].category_name
                    if result.handedness and i < len(result.handedness)
                    else None
                )

                state = get_finger_state(lm, handedness)
                gesture = classify_gesture(state, lm)

                cx, cy = _to_px(lm[0], frame.shape[1], frame.shape[0])
                cv2.putText(
                    frame,
                    f"{gesture} ({handedness})",
                    (cx - 40, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(frame, f"{fps:.1f} FPS", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Multi-Hand Signs (1080p)", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
