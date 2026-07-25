import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(BASE_DIR, 'face_landmarker.task')),
    num_faces=1
)
pose_options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(BASE_DIR, 'pose_landmarker.task')),
    num_poses=1
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def extract_landmarks(video_path, frame_skip=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    face_landmarks_all = []
    pose_landmarks_all = []

    total_frames = 0
    detected_frames = 0
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        total_frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # One bad frame shouldn't abort the whole analysis.
        try:
            face_result = face_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)
        except Exception:
            frame_count += 1
            continue

        if face_result.face_landmarks and pose_result.pose_landmarks:
            detected_frames += 1
            face_landmarks_all.append(face_result.face_landmarks[0])
            pose_landmarks_all.append(pose_result.pose_landmarks[0])

        frame_count += 1

    cap.release()

    detection_ratio = detected_frames / total_frames if total_frames else 0

    return face_landmarks_all, pose_landmarks_all, detection_ratio


def analyze_eye_contact(face_landmarks_all):
    if not face_landmarks_all:
        return 0

    eye_contact_frames = 0
    scored_frames = 0

    for points in face_landmarks_all:
        # The iris points only exist in the 478-landmark output. If a model
        # variant returns the 468-point mesh, skip the frame instead of
        # raising IndexError partway through the video.
        if len(points) <= max(RIGHT_IRIS):
            continue
        scored_frames += 1

        left_iris_x = np.mean([points[i].x for i in LEFT_IRIS])
        right_iris_x = np.mean([points[i].x for i in RIGHT_IRIS])

        left_eye_left = points[LEFT_EYE[0]].x
        left_eye_right = points[LEFT_EYE[1]].x
        right_eye_left = points[RIGHT_EYE[0]].x
        right_eye_right = points[RIGHT_EYE[1]].x

        left_ratio = (left_iris_x - left_eye_left) / (left_eye_right - left_eye_left + 1e-6)
        right_ratio = (right_iris_x - right_eye_left) / (right_eye_right - right_eye_left + 1e-6)

        if 0.35 < left_ratio < 0.65 and 0.35 < right_ratio < 0.65:
            eye_contact_frames += 1

    if not scored_frames:
        return 0.0

    return round(eye_contact_frames / scored_frames, 3)


def analyze_posture(pose_landmarks_all):
    if not pose_landmarks_all:
        return 0

    shoulder_centers = []

    for pose in pose_landmarks_all:
        left = pose[LEFT_SHOULDER]
        right = pose[RIGHT_SHOULDER]

        center_x = (left.x + right.x) / 2
        center_y = (left.y + right.y) / 2

        shoulder_centers.append((center_x, center_y))

    shoulder_centers = np.array(shoulder_centers)

    shoulder_centers[:, 0] = gaussian_filter1d(shoulder_centers[:, 0], sigma=2)
    shoulder_centers[:, 1] = gaussian_filter1d(shoulder_centers[:, 1], sigma=2)

    x_std = np.std(shoulder_centers[:, 0])
    y_std = np.std(shoulder_centers[:, 1])

    movement_score = (x_std * 1.5 + y_std)
    posture_stability = 1 - min(1, movement_score * 4)

    # float(): np.float64 is not JSON-serializable, and this value goes
    # straight into the Flask response.
    return round(float(posture_stability), 3)


def calculate_confidence(eye_contact, posture, detection_ratio):
    score = (eye_contact * 0.5 + posture * 0.3 + detection_ratio * 0.2)
    return round(float(score) * 100, 1)


def analyze_video(video_path):
    face, pose, detection_ratio = extract_landmarks(video_path)

    if not face or not pose:
        return {
            "eye_contact_ratio": 0,
            "posture_stability": 0,
            "confidence_score": 0,
            "detection_ratio": detection_ratio,
            "warning": "Face or pose not detected"
        }

    eye_contact = analyze_eye_contact(face)
    posture = analyze_posture(pose)
    confidence = calculate_confidence(eye_contact, posture, detection_ratio)

    return {
        "eye_contact_ratio": eye_contact,
        "posture_stability": posture,
        "detection_ratio": round(detection_ratio, 3),
        "confidence_score": confidence
    }


if __name__ == "__main__":
    video_path = "test_video.mp4"
    result = analyze_video(video_path)
    print("\n=== HOLISTIC VIDEO ANALYSIS ===")
    for k, v in result.items():
        print(f"{k}: {v}")