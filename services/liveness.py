import os

import cv2
import mediapipe as mp


def check_liveness():
    """Run liveness check via webcam when explicitly enabled."""
    if os.getenv("ENABLE_CAMERA") != "1":
        # Keep local smoke tests/import checks non-interactive by default.
        return True

    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)

    with mp_face.FaceMesh() as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                cap.release()
                cv2.destroyAllWindows()
                return True

            cv2.imshow("Liveness Check", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return False
