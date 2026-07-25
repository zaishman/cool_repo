import cv2
import mediapipe as mp
from mediatest import face_landmarker, pose_landmarker
import subprocess
import shutil
import os

def draw_landmarks_manual(frame, landmarks, color, radius=2):
    h, w = frame.shape[:2]
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), radius, color, -1)


def generate_annotated_video(video_path, output_path, frame_skip=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # write the raw annotated frames to a temp file first - OpenCV's mp4v
    # codec isn't reliably browser-playable, so we transcode afterward
    temp_path = output_path + ".temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    frame_count = 0
    last_face_landmarks = None
    last_pose_landmarks = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            face_result = face_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)

            last_face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None
            last_pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

        annotated = frame

        if last_face_landmarks:
            draw_landmarks_manual(annotated, last_face_landmarks, color=(0, 255, 0), radius=1)
        if last_pose_landmarks:
            draw_landmarks_manual(annotated, last_pose_landmarks, color=(0, 0, 255), radius=3)

        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()

    # transcode to browser-compatible H.264
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", temp_path,
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", output_path],
            capture_output=True,
        )
        os.remove(temp_path)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg transcode failed: {result.stderr.decode(errors='ignore')}")
    else:
        # no ffmpeg available - fall back to the raw (possibly unplayable) file
        os.rename(temp_path, output_path)

    return output_path