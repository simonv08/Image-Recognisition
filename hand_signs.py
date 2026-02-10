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


# MediaPipe Tasks model (HandLandmarker)
# If you don't have it locally, the script will download it next to this file.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model_file(model_path: str) -> None:
    if os.path.exists(model_path):
        return

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Model file not found. Downloading to: {model_path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download the MediaPipe hand_landmarker model. "
            f"Download it manually from {MODEL_URL} and place it at: {model_path}"
        ) from exc


# Connections between the 21 hand landmarks (same topology as the classic Hands solution)
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # Palm
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def _to_pixel(landmark, width: int, height: int) -> Tuple[int, int]:
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def draw_hand_landmarks(
    frame_bgr,
    hand_landmarks: Sequence,
    connections: Iterable[Tuple[int, int]] = HAND_CONNECTIONS,
) -> None:
    height, width = frame_bgr.shape[:2]

    for a, b in connections:
        ax, ay = _to_pixel(hand_landmarks[a], width, height)
        bx, by = _to_pixel(hand_landmarks[b], width, height)
        cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 255, 0), 2)

    for i, lm in enumerate(hand_landmarks):
        x, y = _to_pixel(lm, width, height)
        radius = 6 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame_bgr, (x, y), radius, (0, 0, 255), -1)


@dataclass(frozen=True)
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool


def get_finger_state(hand_landmarks: Sequence, handedness: Optional[str]) -> FingerState:
    # Landmarks indices (MediaPipe hand model)
    THUMB_TIP, THUMB_IP = 4, 3
    INDEX_TIP, INDEX_PIP = 8, 6
    MIDDLE_TIP, MIDDLE_PIP = 12, 10
    RING_TIP, RING_PIP = 16, 14
    PINKY_TIP, PINKY_PIP = 20, 18

    # Heuristic thresholds in normalized coords
    y_margin = 0.02
    x_margin = 0.01

    index_ext = hand_landmarks[INDEX_TIP].y < (hand_landmarks[INDEX_PIP].y - y_margin)
    middle_ext = hand_landmarks[MIDDLE_TIP].y < (hand_landmarks[MIDDLE_PIP].y - y_margin)
    ring_ext = hand_landmarks[RING_TIP].y < (hand_landmarks[RING_PIP].y - y_margin)
    pinky_ext = hand_landmarks[PINKY_TIP].y < (hand_landmarks[PINKY_PIP].y - y_margin)

    # Thumb points sideways; decide based on handedness.
    # If handedness is unknown, we use a looser absolute comparison.
    if handedness == "Right":
        thumb_ext = hand_landmarks[THUMB_TIP].x < (hand_landmarks[THUMB_IP].x - x_margin)
    elif handedness == "Left":
        thumb_ext = hand_landmarks[THUMB_TIP].x > (hand_landmarks[THUMB_IP].x + x_margin)
    else:
        thumb_ext = abs(hand_landmarks[THUMB_TIP].x - hand_landmarks[THUMB_IP].x) > 0.04

    return FingerState(
        thumb=thumb_ext,
        index=index_ext,
        middle=middle_ext,
        ring=ring_ext,
        pinky=pinky_ext,
    )


def _dist2(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def is_thumb_tucked(hand_landmarks: Sequence) -> bool:
    """Heuristic: thumb tip close to the palm/finger bases."""

    thumb_tip = hand_landmarks[4]
    index_mcp = hand_landmarks[5]
    middle_mcp = hand_landmarks[9]

    # Threshold in normalized coords. Tune if needed.
    close_thresh2 = 0.075 * 0.075
    return min(_dist2(thumb_tip, index_mcp), _dist2(thumb_tip, middle_mcp)) < close_thresh2


def _cross(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> Tuple[float, float, float]:
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


def is_palm_facing_camera(hand_landmarks: Sequence, handedness: Optional[str]) -> Optional[bool]:
    """Best-effort palm-facing check.

    Primary heuristic (more robust in practice): use handedness + the thumb being
    on the expected side of the index MCP in the *mirrored/selfie* view.

    Fallback: use the palm normal from (wrist -> left_mcp) x (wrist -> right_mcp),
    where left/right is determined from the current image-space x ordering.

    Returns:
        True  -> palm facing camera
        False -> back of hand facing camera
        None  -> handedness unknown / unreliable
    """

    # Handedness-aware 2D thumb-side heuristic.
    # This code flips the frame horizontally before detection, so the image is a
    # "selfie" mirror view.
    if handedness in ("Left", "Right"):
        thumb_tip = hand_landmarks[4]
        index_mcp = hand_landmarks[5]

        # In mirrored view:
        # - Right hand palm: thumb tends to be on the right side (x greater)
        # - Left hand palm:  thumb tends to be on the left side (x smaller)
        thumb_on_right = thumb_tip.x > index_mcp.x
        if handedness == "Right":
            return thumb_on_right
        return not thumb_on_right

    # wrist + the two outer MCP joints that span the palm
    w = hand_landmarks[0]
    a = hand_landmarks[5]   # index_mcp
    b = hand_landmarks[17]  # pinky_mcp

    # Order by image-space x so the normal sign is consistent for both hands
    # (and stays consistent even if the camera feed is mirrored).
    if a.x > b.x:
        a, b = b, a

    v1x, v1y, v1z = (a.x - w.x), (a.y - w.y), (a.z - w.z)
    v2x, v2y, v2z = (b.x - w.x), (b.y - w.y), (b.z - w.z)
    _, _, nz = _cross(v1x, v1y, v1z, v2x, v2y, v2z)

    # Small magnitude => near edge-on; treat as unknown.
    if abs(nz) < 1e-6:
        return None

    # MediaPipe landmarks use a camera-centric coordinate system where z is
    # typically negative towards the camera. With the consistent ordering above,
    # the normal's z sign can be used to infer which side faces the camera.
    # If this is inverted for a given setup, flip the comparison.
    return nz > 0


def classify_gesture(state: FingerState, hand_landmarks: Sequence, handedness: Optional[str]) -> str:
    extended = (state.thumb, state.index, state.middle, state.ring, state.pinky)

    # If fingers are folded, decide Fist vs Thumbs Up by whether the thumb is tucked.
    others_folded = (not state.index) and (not state.middle) and (not state.ring) and (not state.pinky)
    if others_folded:
        if is_thumb_tucked(hand_landmarks):
            return "Fist"
        return "Thumbs Up"

    # OK sign: thumb tip close to index tip + (typically) other fingers extended
    # Threshold is in normalized coordinates.
    ok_close = _dist2(hand_landmarks[4], hand_landmarks[8]) < (0.055 * 0.055)
    if ok_close and state.middle and state.ring and state.pinky:
        return "OK"

    # Middle finger: middle extended, others folded (thumb can be either)
    if state.middle and (not state.index) and (not state.ring) and (not state.pinky):
        return "Middle Finger"

    # Rock: index + pinky extended, middle + ring folded (thumb can be either)
    if state.index and state.pinky and (not state.middle) and (not state.ring):
        return "Rock"

    # Open hand: treat thumb as optional (thumb extension is often misread depending on pose).
    if state.index and state.middle and state.ring and state.pinky:
        palm_facing = is_palm_facing_camera(hand_landmarks, handedness)
        if palm_facing is True:
            return "Front Palm"
        if palm_facing is False:
            return "Back Hand"
        return "Open Hand"

    if state.index and state.middle and (not state.ring) and (not state.pinky):
        return "Peace"

    return "Unknown"


def main() -> None:
    # Let OpenCV use more CPU threads and optimized kernels.
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    try:
        cpu_threads = os.cpu_count() or 4
        cv2.setNumThreads(int(cpu_threads))
    except Exception:
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "hand_landmarker.task")
    ensure_model_file(model_path)

    # Optional GPU delegate (may not be supported by your installed mediapipe build).
    # Enable with: set MP_USE_GPU=1
    use_gpu = os.getenv("MP_USE_GPU", "0").strip() == "1"
    try:
        delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
        base_options = python.BaseOptions(model_asset_path=model_path, delegate=delegate)
    except Exception:
        base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    # Prefer DirectShow on Windows for lower latency if available.
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0)).")

    # Performance hints (not all cameras/drivers honor these).
    # Many webcams only deliver true 60fps at lower resolutions (e.g. 320x240).
    req_w = int(os.getenv("CAP_WIDTH", "640"))
    req_h = int(os.getenv("CAP_HEIGHT", "480"))
    req_fps = int(os.getenv("CAP_FPS", "60"))

    # Request MJPG first (some drivers decide available modes based on it).
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    cap.set(cv2.CAP_PROP_FPS, req_fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Report actual camera settings (driver may ignore requested values).
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps:.1f}fps (requested {req_w}x{req_h}@{req_fps})")
    if use_gpu:
        print("MediaPipe delegate: GPU (requested)")
    else:
        print("MediaPipe delegate: CPU")

    last_t = time.perf_counter()
    fps = 0.0

    start_perf = time.perf_counter()
    infer_ms = 0.0
    pre_ms = 0.0
    cap_ms = 0.0
    conv_ms = 0.0
    draw_ms = 0.0
    ui_ms = 0.0

    while True:
        t0 = time.perf_counter()
        ok, frame_bgr = cap.read()
        t_cap = time.perf_counter()
        if not ok:
            break

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.perf_counter() - start_perf) * 1000)
        t1 = time.perf_counter()
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        t2 = time.perf_counter()

        gesture = "No hand"
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            handedness = None
            if result.handedness and result.handedness[0]:
                # Categories contain category_name = "Left"/"Right"
                handedness = result.handedness[0][0].category_name

            t_draw0 = time.perf_counter()
            draw_hand_landmarks(frame_bgr, hand_landmarks)
            state = get_finger_state(hand_landmarks, handedness)
            gesture = classify_gesture(state, hand_landmarks, handedness)
            if handedness:
                gesture = f"{gesture} ({handedness})"
            t_draw1 = time.perf_counter()
        else:
            t_draw0 = t_draw1 = time.perf_counter()

        # FPS overlay
        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        infer_ms = 0.9 * infer_ms + 0.1 * ((t2 - t1) * 1000.0)
        pre_ms = 0.9 * pre_ms + 0.1 * ((t1 - t0) * 1000.0)
        cap_ms = 0.9 * cap_ms + 0.1 * ((t_cap - t0) * 1000.0)
        conv_ms = 0.9 * conv_ms + 0.1 * ((t1 - t_cap) * 1000.0)
        draw_ms = 0.9 * draw_ms + 0.1 * ((t_draw1 - t_draw0) * 1000.0)

        cv2.putText(
            frame_bgr,
            f"Gesture: {gesture}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"FPS: {fps:.1f}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Cap: {cap_ms:.1f}ms  Conv: {conv_ms:.1f}ms",
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Infer: {infer_ms:.1f}ms  Draw: {draw_ms:.1f}ms  UI: {ui_ms:.1f}ms",
            (10, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        t_ui0 = time.perf_counter()
        cv2.imshow("Hand Sign Recognition", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        t_ui1 = time.perf_counter()
        ui_ms = 0.9 * ui_ms + 0.1 * ((t_ui1 - t_ui0) * 1000.0)

        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
