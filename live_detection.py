import cv2
import mediapipe as mp
import joblib
import time
from collections import deque
from utils import compute_features

model = joblib.load("models/gait_model.pkl")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === GAIT TRACKING VARIABLES ===
prev_left_heel_y = None
prev_right_heel_y = None

left_heel_strikes = []
right_heel_strikes = []

last_left_strike_time = None
last_right_strike_time = None

cadence = 0
stride_time = 0

start_time = time.time()

cap = cv2.VideoCapture(0)

prev_hip = None
prev_velocity = None

prediction_buffer = deque(maxlen=10)


# ==============================
# STABILITY SCORE FUNCTION
# ==============================

def compute_stability_score(shoulder_tilt, hip_tilt, velocity, acceleration):

    score = 100

    score -= abs(shoulder_tilt) * 0.5
    score -= abs(hip_tilt) * 0.5
    score -= velocity * 50
    score -= acceleration * 30

    score = max(0, min(100, score))

    return score


while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # === HEEL LANDMARKS ===
        left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]

        current_left_heel_y = left_heel.y
        current_right_heel_y = right_heel.y
        current_time = time.time()

        # === LEFT HEEL STRIKE ===
        if prev_left_heel_y is not None:

            if abs(prev_left_heel_y - current_left_heel_y) < 0.003:

                left_heel_strikes.append(current_time)

                if last_left_strike_time is not None:
                    stride_time = current_time - last_left_strike_time

                last_left_strike_time = current_time

        prev_left_heel_y = current_left_heel_y

        # === RIGHT HEEL STRIKE ===
        if prev_right_heel_y is not None:

            if abs(prev_right_heel_y - current_right_heel_y) < 0.003:

                right_heel_strikes.append(current_time)

                if last_right_strike_time is not None:
                    stride_time = current_time - last_right_strike_time

                last_right_strike_time = current_time

        prev_right_heel_y = current_right_heel_y

        # === CADENCE ===
        total_steps = len(left_heel_strikes) + len(right_heel_strikes)

        elapsed_time = current_time - start_time

        if elapsed_time > 0:
            cadence = (total_steps / elapsed_time) * 60

        # === FEATURE EXTRACTION ===
        keypoints = [(lm.x, lm.y) for lm in landmarks]

        shoulder_tilt, hip_tilt, step_length, velocity, acceleration, hip_center = compute_features(
            keypoints, prev_hip, prev_velocity
        )

        # ==============================
        # STABILITY SCORE
        # ==============================

        stability_score = compute_stability_score(
            shoulder_tilt,
            hip_tilt,
            velocity,
            acceleration
        )

        # ==============================
        # MODEL PREDICTION
        # ==============================

        prediction = model.predict(
            [[shoulder_tilt, hip_tilt, step_length, velocity, acceleration]]
        )[0]

        prediction_buffer.append(prediction)

        final_prediction = max(set(prediction_buffer), key=prediction_buffer.count)

        # ==============================
        # STATUS BASED ON STABILITY SCORE
        # ==============================

        if stability_score > 80:

            status = "STABLE"
            color = (0, 255, 0)

        elif stability_score > 60:

            status = "MILDLY UNSTABLE"
            color = (0, 165, 255)

        else:

            status = "UNSTABLE - FALL RISK"
            color = (0, 0, 255)

        # ==============================
        # DEVIATION GUIDANCE
        # ==============================

        if shoulder_tilt > 5:

            guidance = f"Move {abs(shoulder_tilt):.1f}° LEFT to stabilize"

        elif shoulder_tilt < -5:

            guidance = f"Move {abs(shoulder_tilt):.1f}° RIGHT to stabilize"

        else:

            guidance = "Good Alignment"

        # ==============================
        # DISPLAY TEXT
        # ==============================

        cv2.putText(frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        cv2.putText(frame, f"Shoulder Tilt: {shoulder_tilt:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Hip Tilt: {hip_tilt:.2f}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Step Length: {step_length:.4f}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Velocity: {velocity:.4f}", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Acceleration: {acceleration:.4f}", (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Cadence: {cadence:.2f} steps/min",
                    (30, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 2)

        cv2.putText(frame, f"Stride Time: {stride_time:.2f} sec",
                    (30, 270), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 2)

        cv2.putText(frame, f"Stability Score: {stability_score:.1f}/100",
                    (30, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        cv2.putText(frame, guidance,
                    (30, 330), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        prev_velocity = velocity
        prev_hip = hip_center

        # ==============================
        # COLORED SKELETON
        # ==============================

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=color, thickness=3, circle_radius=3),
            mp_drawing.DrawingSpec(color=color, thickness=2)
        )

    cv2.imshow("Gait Stability Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()