import os
import time
import math
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Automatically downloads the open-source MediaPipe hand model on the first run
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Connect to Windows Master Volume
devices = AudioUtilities.GetSpeakers()
if hasattr(devices, "EndpointVolume"):
    volume = devices.EndpointVolume
else:
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)

# Set up MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_ts = 0
vol_percent = int(volume.GetMasterVolumeLevelScalar() * 100)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    ts = max(int(time.time() * 1000), last_ts + 1)
    last_ts = ts
    result = detector.detect_for_video(mp_image, ts)

    is_locked = False

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        # Check if hand is closed into a fist (Index & Middle tips curled below middle joints)
        index_closed = lm[8].y > lm[6].y
        middle_closed = lm[12].y > lm[10].y
        is_locked = index_closed and middle_closed

        if not is_locked:
            # Thumb tip (4), Index tip (8), Wrist (0), Middle knuckle (9)
            x1, y1 = int(lm[4].x * w), int(lm[4].y * h)
            x2, y2 = int(lm[8].x * w), int(lm[8].y * h)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            wx, wy = int(lm[0].x * w), int(lm[0].y * h)
            mx, my = int(lm[9].x * w), int(lm[9].y * h)

            # Normalize pinch distance by palm size
            pinch_dist = math.hypot(x2 - x1, y2 - y1)
            palm_size = max(math.hypot(mx - wx, my - wy), 1.0)
            ratio = pinch_dist / palm_size

            # Map pinch ratio to 0% - 100% volume
            vol_scalar = float(np.interp(ratio, [0.20, 1.15], [0.0, 1.0]))
            vol_percent = int(round(vol_scalar * 50) * 2)
            volume.SetMasterVolumeLevelScalar(vol_percent / 100.0, None)

            # Visual pinch line and dots
            cv2.circle(frame, (x1, y1), 9, (255, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 9, (255, 0, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            mid_color = (0, 255, 0) if vol_percent == 0 else (255, 0, 255)
            cv2.circle(frame, (cx, cy), 8, mid_color, -1)

    # Draw Volume Bar HUD (Yellow when locked, Green when active)
    bar_color = (0, 200, 255) if is_locked else (0, 230, 0)
    bar_top, bar_bottom = 120, 380
    bar_fill = int(np.interp(vol_percent, [0, 100], [bar_bottom, bar_top]))
    cv2.rectangle(frame, (40, bar_top), (75, bar_bottom), (255, 255, 255), 2)
    cv2.rectangle(frame, (42, bar_fill), (73, bar_bottom - 2), bar_color, -1)

    status_label = f"LOCKED: {vol_percent}%" if is_locked else f"Vol: {vol_percent}%"
    cv2.putText(frame, status_label, (25, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, bar_color, 2)
    cv2.putText(frame, "Pinch: Adjust | Fist: Lock | [Q] Quit", (25, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    cv2.imshow("WaveCtrl", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()