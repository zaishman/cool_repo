import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'face_landmarker.task')
base_options = mp_python.BaseOptions(model_asset_path=model_path)

options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

def extract_face_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    all_frame_landmarks = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            all_frame_landmarks.append(result.face_landmarks[0])

    cap.release()
    return all_frame_landmarks

LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
NOSE_TIP = 1
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263

def analyse_eye_contact_and_posture(all_frame_landmarks, frame_width, frame_height):
    eye_contact_frames = 0
    head_positions = []

    for landmarks in all_frame_landmarks:
        points = landmarks

        left_iris_x = np.mean([points[i].x for i in LEFT_IRIS])
        right_iris_x = np.mean([points[i].x for i in RIGHT_IRIS])
        left_corner_x = points[LEFT_EYE_CORNER].x
        right_corner_x = points[RIGHT_EYE_CORNER].x

        eye_center_ratio = (left_iris_x + right_iris_x) / 2
        face_center_ratio = (left_corner_x + right_corner_x) / 2

        if abs(eye_center_ratio - face_center_ratio) < 0.008:  # threshold for "looking forward"
            eye_contact_frames += 1

        nose = points[NOSE_TIP]
        head_positions.append((nose.x, nose.y))

    eye_contact_ratio = eye_contact_frames / len(all_frame_landmarks) if all_frame_landmarks else 0

    head_positions = np.array(head_positions)

    raw_std = np.std(head_positions)
    print(f"raw std: {raw_std}")  # add this temporarily
    posture_stability = 1 - min(1, np.std(head_positions) * 3)

    return {
        "eye_contact_ratio": round(eye_contact_ratio, 3),
        "posture_stability": round(posture_stability, 3)
    }

def calculate_video_confidence_score(eye_contact_ratio, posture_stability):
    confidence_score = (eye_contact_ratio * 0.6) + (posture_stability * 0.4)
    return round(confidence_score * 100, 1)

def analyze_video(video_path):
    landmarks = extract_face_landmarks(video_path)

    if not landmarks:
        return {
            "eye_contact_ratio": 0,
            "posture_stability": 0,
            "confidence_score": 0,
            "warning": "No face detected in video"
        }

    eye_posture_result = analyse_eye_contact_and_posture(landmarks, None, None)
    confidence_score = calculate_video_confidence_score(
        eye_posture_result["eye_contact_ratio"],
        eye_posture_result["posture_stability"]
    )

    return {
        "eye_contact_ratio": eye_posture_result["eye_contact_ratio"],
        "posture_stability": eye_posture_result["posture_stability"],
        "confidence_score": confidence_score
    }