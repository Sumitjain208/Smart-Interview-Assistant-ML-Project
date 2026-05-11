import cv2
import mediapipe as mp
def webcam_score():
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection()
    cap = cv2.VideoCapture(0)
    score = 0

    for i in range(50):
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_detection.process(rgb_frame)
        if results.detections:
            score += 1

    cap.release()
    cv2.destroyAllWindows()

    return score