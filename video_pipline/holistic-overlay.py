import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def generate_annotated_video(video_path, output_path, show_live=True):
    """
    Reads video_path, runs MediaPipe Holistic (face mesh + body pose) on
    every frame, draws the landmarks directly onto that frame, writes the
    annotated result to output_path, and optionally displays it live in a
    window while processing.

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
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            writer.write(annotated)

            if show_live:
                cv2.imshow("MediaPipe Holistic - press Q to stop early", annotated)
                # waitKey(1) keeps playback moving at roughly video speed;
                # pressing 'q' stops processing early
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    writer.release()
    if show_live:
        cv2.destroyAllWindows()

    return output_path
