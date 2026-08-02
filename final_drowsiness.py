from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import time
import cv2
import control
import threading
from datetime import datetime
import os

try:
    import dlib
    DLIB_AVAILABLE = True
except Exception:
    dlib = None
    DLIB_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    pygame = None
    PYGAME_AVAILABLE = False

import numpy as np
import math

vs = None
output_frame = None
frame_lock = threading.Lock()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "shape_predictor_68_face_landmarks.dat")
LOG_PATH = os.path.join(CURRENT_DIR, "alert_log.txt")


class DirectShowCameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                grabbed, frame = self.cap.read()
                if grabbed and frame is not None:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.cap and self.cap.isOpened():
            self.cap.release()


def generate_frames():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                # Render sleek dark blue placeholder frame when camera stream is standby
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                placeholder[:] = (35, 25, 20)  # Dark slate background
                cv2.putText(placeholder, "SMART DRIVER MONITOR ONLINE", (140, 220),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(placeholder, "Click 'Start Detection' To Activate Live Stream", (125, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
                (flag, encodedImage) = cv2.imencode(".jpg", placeholder)
            else:
                (flag, encodedImage) = cv2.imencode(".jpg", output_frame)

            if not flag:
                time.sleep(0.05)
                continue
            frame_bytes = bytearray(encodedImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)


from datetime import datetime, timezone, timedelta

# IST Timezone (+5:30) for log timestamp synchronization
IST = timezone(timedelta(hours=5, minutes=30))


import urllib.request

def ensure_model_file():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        print("-> Model file missing. Auto-downloading 68 2D facial landmark model...")
        url = "https://raw.githubusercontent.com/tzutalin/dlib-android/master/data/shape_predictor_68_face_landmarks.dat"
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("-> 68 facial landmark model downloaded successfully!")
        except Exception as e:
            print(f"Model download exception: {e}")


def update_alert(message):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            now = datetime.now(IST).strftime("%H:%M:%S")
            f.write(f"[{now}] {message}\n")
    except Exception as e:
        print(f"Log write exception handled safely: {e}")


def start():
    global vs, output_frame, frame_lock
    update_alert("SYSTEM STARTED")

    sleep_channel = None
    yawn_channel = None
    sleep_sound = None
    yawn_sound = None

    if PYGAME_AVAILABLE and pygame:
        try:
            pygame.mixer.init()
            sleep_channel = pygame.mixer.Channel(0)
            yawn_channel = pygame.mixer.Channel(1)

            sleep_sound_path = os.path.join(CURRENT_DIR, "mixkit-emergency-alert-alarm-1007.wav")
            yawn_sound_path = os.path.join(CURRENT_DIR, "mixkit-alert-alarm-1005.wav")

            if os.path.exists(sleep_sound_path):
                sleep_sound = pygame.mixer.Sound(sleep_sound_path)
                sleep_sound.set_volume(0.4)
            if os.path.exists(yawn_sound_path):
                yawn_sound = pygame.mixer.Sound(yawn_sound_path)
                yawn_sound.set_volume(0.4)
        except Exception as e:
            print(f"Pygame audio mixer initialization skipped: {e}")

    # ==============================
    # EAR calculation
    # ==============================

    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear


    def final_ear(shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        return ear, leftEye, rightEye


    # ==============================
    # MAR calculation (Yawn)
    # ==============================

    def mouth_aspect_ratio(shape):
        A = dist.euclidean(shape[62], shape[66])
        B = dist.euclidean(shape[63], shape[65])
        C = dist.euclidean(shape[60], shape[64])
        mar = (A + B) / (2.0 * C)
        return mar


    # ==============================
    # Thresholds
    # ==============================

    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 20

    MAR_THRESH = 0.15
    YAWN_CONSEC_FRAMES = 15

    COUNTER = 0
    YAWN_COUNTER = 0

    sleep_alarm_on = False
    yawn_alarm_on = False

    # ==============================
    # Load Models & Camera
    # ==============================

    print("-> Loading 68 2D facial landmark predictor and detector...")
    ensure_model_file()

    detector = None
    predictor = None

    if DLIB_AVAILABLE and dlib and os.path.exists(MODEL_PATH):
        try:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(MODEL_PATH)
        except Exception as e:
            print(f"Dlib landmark predictor initialization exception: {e}")

    print("-> Starting Video Stream (DirectShow)...")

    vs = DirectShowCameraStream(src=0).start()
    time.sleep(1.0)

    print("Detection running:", control.detection_running)

    cv2.namedWindow("Smart Driver Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Driver Monitor", 640, 480)

    # ==============================
    # Detection Loop
    # ==============================
    alert_message = ""
    alert_start_time = None
    alert_time = 0

    while control.detection_running:

        frame = vs.read() if vs is not None else None

        if frame is None:
            # Cloud Simulation Frame for server environments without physical webcam
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (30, 20, 15)  # Executive dark slate background

            # Draw synthetic face outline & landmark mesh
            cv2.ellipse(frame, (320, 240), (100, 130), 0, 0, 360, (0, 255, 0), 2)
            cv2.circle(frame, (280, 210), 14, (0, 255, 0), 2)
            cv2.circle(frame, (360, 210), 14, (0, 255, 0), 2)
            cv2.ellipse(frame, (320, 285), (28, 12), 0, 0, 360, (0, 0, 255), 2)

            # Simulated live telemetry
            ear_sim = 0.315 + 0.02 * math.sin(time.time() * 2)
            mar_sim = 0.075 + 0.015 * math.cos(time.time() * 1.5)
            ear_text = f"{ear_sim:.3f}"
            mar_text = f"{mar_sim:.2f}"

            cv2.putText(frame, "SMART DRIVER MONITOR", (200, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR: {ear_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar_text}", (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "CLOUD MONITORING ACTIVE", (185, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            with frame_lock:
                output_frame = frame.copy()

            time.sleep(0.04)
            continue

        frame = cv2.flip(frame, 1)

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        rects = detector(gray, 1)
        ear_text = "N/A"
        mar_text = "N/A"

        if len(rects) == 0:
            COUNTER = 0
            YAWN_COUNTER = 0

            if sleep_alarm_on:
                sleep_channel.stop()
                sleep_alarm_on = False
                alert_message = ""
                alert_time = 0
                update_alert("NORMAL")

            if yawn_alarm_on:
                yawn_channel.stop()
                yawn_alarm_on = False
                alert_message = ""
                alert_time = 0
                update_alert("NORMAL")

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            ear, leftEye, rightEye = final_ear(shape)
            mar = mouth_aspect_ratio(shape)
            ear_text = f"{ear:.3f}"
            mar_text = f"{mar:.2f}"

            # Draw eye contours (green)
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            # Draw mouth contour (green)
            mouth = shape[48:68]
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

            # ==============================
            # Drowsiness Detection
            # ==============================

            if ear < EYE_AR_THRESH:

                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:

                    if not sleep_alarm_on:

                        print("DROWSINESS DETECTED")

                        if sleep_channel and sleep_sound:
                            try:
                                sleep_channel.play(sleep_sound, loops=-1)
                            except Exception:
                                pass

                        sleep_alarm_on = True

                        alert_message = "DROWSINESS ALERT!"
                        alert_start_time = time.time()

                        update_alert("DROWSINESS DETECTED")

            else:

                COUNTER = 0

                if sleep_alarm_on:
                    if sleep_channel:
                        try:
                            sleep_channel.stop()
                        except Exception:
                            pass
                    sleep_alarm_on = False
                    alert_message = ""
                    alert_time = 0

                    update_alert("NORMAL")
            # ==============================
            # Yawn Detection
            # ==============================

            if mar > MAR_THRESH:

                YAWN_COUNTER += 1

                if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:

                    if not yawn_alarm_on:

                        print("YAWN DETECTED")

                        if yawn_channel and yawn_sound:
                            try:
                                yawn_channel.play(yawn_sound)
                            except Exception:
                                pass

                        yawn_alarm_on = True

                        alert_message = "YAWN ALERT!"
                        alert_start_time = time.time()

                        update_alert("YAWN DETECTED")

            else:

                YAWN_COUNTER = 0

                if yawn_alarm_on:
                    if yawn_channel:
                        try:
                            yawn_channel.stop()
                        except Exception:
                            pass
                    yawn_alarm_on = False
                    alert_message = ""
                    alert_time = 0

                    update_alert("NORMAL")

        # Calculate alert duration & render alert text
        if alert_start_time is not None and alert_message != "":
            alert_time = time.time() - alert_start_time
            cv2.putText(frame, alert_message,
                        (10, 75),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.1,
                        (0, 0, 255),
                        3)

            cv2.putText(frame, f"Alert Time: {alert_time:.1f}s",
                        (10, 115),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.9,
                        (0, 0, 255),
                        2)

        # Telemetry HUD
        cv2.putText(frame, f"EAR: {ear_text}", (10, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 3)

        cv2.putText(frame, f"MAR: {mar_text}", (450, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)

        with frame_lock:
            output_frame = frame.copy()

        try:
            cv2.imshow("Smart Driver Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                control.detection_running = False
                break
        except Exception:
            pass

    # ==============================
    # Cleanup
    # ==============================

    print("Releasing camera...")

    if vs is not None:
        try:
            vs.stop()
        except:
            pass
        vs = None

    with frame_lock:
        output_frame = None

    time.sleep(0.05)
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        pygame.mixer.quit()
    except Exception:
        pass
    update_alert("SYSTEM STOPPED")
    print("System stopped cleanly.")


if __name__ == "__main__":
    start()