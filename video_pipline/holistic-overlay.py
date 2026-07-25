import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def generate_annotated_video(video_path, output_path):
    """
    Reads video_path, runs MediaPipe Holistic (face mesh + body pose + both
    hands) on every frame, draws the landmarks directly onto that frame, and
    writes the annotated result to output_path so it can be played back.

    Returns output_path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
    ) as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True

            annotated = frame  # draw straight onto the original BGR frame

            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
