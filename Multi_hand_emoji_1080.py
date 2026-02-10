from __future__ import annotations
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, List

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# ============================================================
# CONFIG
# ============================================================

CAP_W, CAP_H, CAP_FPS = 1920, 1080, 60
INF_W, INF_H = 640, 360
DRAW_DEBUG = True
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
GESTURE_HOLD_TIME = 1.0       # seconds to activate emoji spam
EMOJI_SPAWN_RATE = 0.75      # seconds between emoji bursts per held gesture
EMOJI_BURST_COUNT = 2       # emojis per burst
EMOJI_RISE_SPEED_MIN = -320.0  # pixels/sec (more negative = faster up)
EMOJI_RISE_SPEED_MAX = -180.0  # pixels/sec
EMOJI_COLOR = "#00ffff"        # cyan. Use "#ff00ff" for magenta
EMOJI_FADE_SPEED = 5.0        # alpha fade per second
EMOJIS = {
    "Thumbs Up": "👍",
    "Fist": "✊",
    "OK": "👌",
    "Peace": "✌️",
    "Rock": "🤘",
    "Middle Finger": "🖕",
    "Open Hand": "🖐️",
}

# ============================================================
# UTILS
# ============================================================

def ensure_model_file(path: str) -> None:
    if os.path.exists(path):
        return
    print("Downloading MediaPipe hand_landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, path)


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
    return dx*dx + dy*dy

def classify_gesture(state: FingerState, lm) -> str:
    # Make Fist detection less sensitive for thumbs up
    if not state.index and not state.middle and not state.ring and not state.pinky:
        if state.thumb:
            return "Thumbs Up"
        return "Fist"
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
# EMOJI SMOKE
# ============================================================

@dataclass
class EmojiParticle:
    text: str
    x: float
    y: float
    vy: float
    alpha: float

def draw_unicode_text(frame, text, x, y, font_size=48, alpha=1.0):
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        s = hex_color.strip().lstrip("#")
        if len(s) != 6:
            raise ValueError("Expected 6 hex digits")
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return r, g, b

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    txt_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_layer)
    try:
        font = ImageFont.truetype("seguiemj.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    try:
        r, g, b = _hex_to_rgb(EMOJI_COLOR)
    except Exception:
        r, g, b = 255, 255, 255
    txt_draw.text((x, y), text, font=font, fill=(r, g, b, int(alpha*255)))
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), txt_layer)
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

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
        num_hands=2,
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

    print(f"Camera: {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.1f} FPS")

    fps = 0.0
    last_t = time.perf_counter()
    start_t = time.perf_counter()

    # Track the current gesture per hand and how long it's been held.
    gesture_state: dict[int, tuple[str, float]] = {}  # hand_index -> (gesture, gesture_start_time)
    # Throttle emoji spawning so we don't spawn every frame.
    last_spawn: dict[tuple[int, str], float] = {}      # (hand_index, gesture) -> last_spawn_time
    emoji_particles: List[EmojiParticle] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # --- Inference ---
        inf = cv2.resize(frame, (INF_W, INF_H), interpolation=cv2.INTER_LINEAR)
        inf_rgb = cv2.cvtColor(inf, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, inf_rgb)
        ts_ms = int((time.perf_counter() - start_t)*1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        hands_count = len(result.hand_landmarks) if result.hand_landmarks else 0

        for i in range(hands_count):
            lm = result.hand_landmarks[i]
            if DRAW_DEBUG:
                draw_hand_landmarks(frame, lm)
            handedness = (result.handedness[i][0].category_name
                          if result.handedness and result.handedness[i] else None)
            state = get_finger_state(lm, handedness)
            gesture = classify_gesture(state, lm)

            # Track gesture hold time (only for the current gesture)
            now_t = time.perf_counter()
            prev = gesture_state.get(i)
            if prev is None or prev[0] != gesture:
                gesture_state[i] = (gesture, now_t)
                hold_time = 0.0
            else:
                hold_time = now_t - prev[1]

            # Spawn emojis if held long enough
            if hold_time >= GESTURE_HOLD_TIME:
                # Throttle bursts so we don't spawn every frame.
                key = (i, gesture)
                last_t_spawn = last_spawn.get(key, 0.0)
                if (now_t - last_t_spawn) >= EMOJI_SPAWN_RATE:
                    last_spawn[key] = now_t
                    center_x, center_y = _to_px(lm[9], frame.shape[1], frame.shape[0])
                    for _ in range(EMOJI_BURST_COUNT):
                        emoji_particles.append(
                            EmojiParticle(
                                text=EMOJIS.get(gesture, "❓"),
                                x=center_x + random.uniform(-20, 20),
                                y=center_y + random.uniform(-20, 20),
                                vy=random.uniform(EMOJI_RISE_SPEED_MIN, EMOJI_RISE_SPEED_MAX),
                                alpha=1.0,
                            )
                        )

        # Update and draw emoji particles
        dt = time.perf_counter() - last_t
        new_particles = []
        for p in emoji_particles:
            p.y += p.vy * dt
            p.alpha -= EMOJI_FADE_SPEED * dt
            if p.alpha > 0:
                frame = draw_unicode_text(frame, p.text, p.x, p.y, font_size=48, alpha=p.alpha)
                new_particles.append(p)
        emoji_particles = new_particles

        # FPS overlay
        now = time.perf_counter()
        dt_fps = now - last_t
        last_t = now
        if dt_fps > 0:
            fps = 0.9*fps + 0.1*(1.0/dt_fps)

        cv2.putText(frame, f"{fps:.1f} FPS", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow("Hand Emoji Smoke", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
