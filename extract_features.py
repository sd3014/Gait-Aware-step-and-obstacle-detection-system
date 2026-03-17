import cv2
import os
import pandas as pd
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

dataset_root = "dataset/ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"

data = []

WINDOW = 10

for subject in os.listdir(dataset_root):

    subject_path = os.path.join(dataset_root, subject)

    if not os.path.isdir(subject_path):
        continue

    for category in ["ADL","Fall"]:

        category_path = os.path.join(subject_path, category)

        if not os.path.exists(category_path):
            continue

        print("Processing:",subject,category)

        for video in os.listdir(category_path):

            video_path = os.path.join(category_path, video)

            cap = cv2.VideoCapture(video_path)

            pose_seq = []

            while True:

                ret,frame = cap.read()

                if not ret:
                    break

                results = pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:

                    keypoints = []

                    for lm in results.pose_landmarks.landmark:
                        keypoints.append(lm.x)
                        keypoints.append(lm.y)

                    pose_seq.append(keypoints)

            cap.release()

            pose_seq = np.array(pose_seq)

            if len(pose_seq) < WINDOW:
                continue

            for i in range(len(pose_seq)-WINDOW):

                segment = pose_seq[i:i+WINDOW]

                mean_pose = np.mean(segment,axis=0)

                velocity = np.mean(np.diff(segment,axis=0))

                acceleration = np.mean(np.diff(segment,axis=0)**2)

                pose_variance = np.var(segment)

                features = list(mean_pose) + [
                    velocity,
                    acceleration,
                    pose_variance
                ]

                label = 1 if category=="Fall" else 0

                data.append(features+[label,subject])


columns = []

for i in range(33):
    columns.append(f"x{i}")
    columns.append(f"y{i}")

columns += [
    "velocity",
    "acceleration",
    "pose_variance",
    "label",
    "subject"
]

df = pd.DataFrame(data,columns=columns)

df.to_csv("gait_features.csv",index=False)

print("Feature extraction complete")
print("Total samples:",len(df))