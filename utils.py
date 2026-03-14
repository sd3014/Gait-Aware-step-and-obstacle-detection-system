import numpy as np

def calculate_angle(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_features(keypoints, prev_hip_center, prev_velocity):

    shoulder_tilt = calculate_angle(keypoints[11], keypoints[12])
    hip_tilt = calculate_angle(keypoints[23], keypoints[24])
    step_length = euclidean(keypoints[27], keypoints[28])

    hip_center = (
        (keypoints[23][0] + keypoints[24][0]) / 2,
        (keypoints[23][1] + keypoints[24][1]) / 2
    )

    velocity = 0
    acceleration = 0

    if prev_hip_center is not None:
        velocity = euclidean(hip_center, prev_hip_center)

    if prev_velocity is not None:
        acceleration = velocity - prev_velocity

    return shoulder_tilt, hip_tilt, step_length, velocity, acceleration, hip_center