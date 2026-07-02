from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import pygame
import control
from datetime import datetime
vs=None
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "shape_predictor_68_face_landmarks.dat")
def update_alert(message):

    with open("alert_log.txt", "a") as f:

        now = datetime.now().strftime("%H:%M:%S")

        f.write(f"[{now}] {message}\n")


def start():
    global vs
    update_alert("SYSTEM STARTED")

    pygame.mixer.init()
    sleep_channel = pygame.mixer.Channel(0)
    yawn_channel = pygame.mixer.Channel(1)

    # Alarm sounds
    sleep_sound = pygame.mixer.Sound("mixkit-emergency-alert-alarm-1007.wav")
    yawn_sound = pygame.mixer.Sound("mixkit-alert-alarm-1005.wav")

    sleep_sound.set_volume(0.4)
    yawn_sound.set_volume(0.4)

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
    # Load Models
    # ==============================

    

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    print("-> Loading predictor and detector...")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)

    print("-> Starting Video Stream...")

    vs = VideoStream(src=0, usePiCamera=False).start()
    time.sleep(1.0)

    print("Detection running:", control.detection_running)

    # ==============================
    # Detection Loop
    # ==============================
    # Display message variables
    alert_message = ""
    alert_start_time = None
    alert_time = 0
    while control.detection_running:
        

        frame = vs.read()
        frame = cv2.flip(frame, 1)

        if frame is None:
            time.sleep(0.01)
            continue

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        rects = detector(gray, 1)
        for rect in rects:

            (x1, y1, x2, y2) = (rect.left(), rect.top(), rect.right(), rect.bottom())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # draw landmarks (optional but useful)
            #for (x, y) in shape:
                #cv2.circle(frame, (x, y), 2, (0,255,255), -1)
            ear, leftEye, rightEye = final_ear(shape)
            mar = mouth_aspect_ratio(shape)

            # Draw eyes
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0,255,0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,255,0), 1)

            # Draw mouth
            mouth = shape[48:68]
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255,0,0), 1)

                    # ==============================
            # Drowsiness Detection
            # ==============================

            if ear < EYE_AR_THRESH:

                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:

                    if not sleep_alarm_on:

                        print("DROWSINESS DETECTED")

                        sleep_channel.play(sleep_sound, loops=-1)
                        sleep_alarm_on = True

                        alert_message = "DROWSINESS ALERT!"
                        alert_start_time = time.time()

                        update_alert("DROWSINESS DETECTED")

            else:

                COUNTER = 0

                if sleep_alarm_on:
                    sleep_channel.stop()
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

                        yawn_channel.play(yawn_sound)

                        yawn_alarm_on = True

                        alert_message = "YAWN ALERT!"
                        alert_start_time = time.time()

                        update_alert("YAWN DETECTED")

            else:

                YAWN_COUNTER = 0

                if yawn_alarm_on:
                    yawn_channel.stop()
                    yawn_alarm_on = False
                    alert_message = ""
                    alert_time = 0

                    update_alert("NORMAL")
            # Calculate alert duration
            if alert_start_time is not None:
                alert_time = time.time() - alert_start_time
                        # ==============================
                        # Alert message display
            if alert_message != "":
                cv2.putText(frame, alert_message,
                            (150, 80),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.2,
                            (0,0,255),
                            3)

                cv2.putText(frame, f"Alert Time: {alert_time:.1f}s",
                            (170,120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,0,255),
                            2)
                        # Display
            # ==============================

            cv2.putText(frame,"SMART DRIVER MONITOR",(200,30),
            cv2.FONT_HERSHEY_DUPLEX,0.7,(255,255,255),2)

            cv2.putText(frame,f"EAR: {ear:.3f}",(10,30),
            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            cv2.putText(frame,f"MAR: {mar:.2f}",(480,30),
            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)


        cv2.imshow("Smart Driver Monitor", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            control.detection_running = False
            break


    # ==============================
    # Cleanup
    # ==============================

    print("Releasing camera...")

    if vs is not None:
        try:
            vs.stream.release()
        except:
            pass

        try:
            vs.stop()
        except:
            pass

        vs = None

    time.sleep(1)

    cv2.destroyAllWindows()

    pygame.mixer.quit()

    update_alert("SYSTEM STOPPED")

    print("System stopped cleanly.")


if __name__ == "__main__":
    start()