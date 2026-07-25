# test_video.py
from mediapipe_analysis import extract_face_landmarks, analyse_eye_contact_and_posture

landmarks = extract_face_landmarks("video2.mov")
print(f"Frames with detected faces: {len(landmarks)}")

result = analyse_eye_contact_and_posture(landmarks, frame_width=None, frame_height=None)
print(result)