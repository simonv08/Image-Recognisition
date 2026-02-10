from __future__ import annotations
import cv2
import time
import random
import numpy as np

# -------------------------------
# CONFIG
# -------------------------------
WINDOW_W, WINDOW_H = 1280, 720
FPS = 60

EMOJI_SIZE = 64          # pixels
EMOJI_LIFETIME = 2.0     # seconds
GESTURE_HOLD_TIME = 1.0  # seconds before spawning emoji

# Map gestures to emojis
GESTURE_EMOJIS = {
    "Thumbs Up": "👍",
    "Fist": "✊",
    "OK": "👌",
    "Peace": "✌️",
    "Rock": "🤘",
    "Middle Finger": "🖕",
    "Open Hand": "🖐️",
}

# -------------------------------
# Emoji Object
# -------------------------------
class Emoji:
    def __init__(self, text, x, y):
        self.text = text
        self.x = x
        self.y = y
        self.start_time = time.perf_counter()

    def draw(self, frame):
        age = time.perf_counter() - self.start_time
        alpha = max(0, 1 - age / EMOJI_LIFETIME)
        if alpha <= 0:
            return False  # expired

        # Create overlay for fading
        overlay = frame.copy()
        font_scale = EMOJI_SIZE / 64
        cv2.putText(
            overlay,
            self.text,
            (int(self.x), int(self.y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # Move up slowly
        self.y -= 1
        return True

# -------------------------------
# MAIN LOOP
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

    emojis: list[Emoji] = []
    last_gesture = "No hand"
    last_change_time = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # -------------------------------
        # Simulate hand gesture detection
        # Replace this with your real gesture
        # -------------------------------
        # For demo: cycle through gestures randomly every few seconds
        if random.random() < 0.01:
            last_gesture = random.choice(list(GESTURE_EMOJIS.keys()))
            last_change_time = time.perf_counter()

        # -------------------------------
        # Spawn emoji if held > GESTURE_HOLD_TIME
        # -------------------------------
        if last_gesture != "No hand":
            held_time = time.perf_counter() - last_change_time
            if held_time > GESTURE_HOLD_TIME:
                # Spawn at random x near center bottom
                x = random.randint(WINDOW_W//3, WINDOW_W*2//3)
                y = WINDOW_H - 50
                emojis.append(Emoji(GESTURE_EMOJIS[last_gesture], x, y))
                last_change_time = time.perf_counter()  # reset for next spawn

        # -------------------------------
        # Draw all emojis
        # -------------------------------
        emojis = [e for e in emojis if e.draw(frame)]

        # -------------------------------
        # Draw HUD
        # -------------------------------
        cv2.putText(frame, f"Gesture: {last_gesture}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Emoji Spam Demo", frame)
        if cv2.waitKey(int(1000 / FPS)) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
