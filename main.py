import cv2
import mediapipe as mp
import numpy as np
import transforms3d

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)


def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def compute_orientation(landmarks, image_shape):
    nose_tip = np.array([landmarks[4].x, landmarks[4].y])
    left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
    right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])

    eye_line = right_eye_inner - left_eye_inner
    nose_to_eye_line = nose_tip - left_eye_inner

    eye_line_norm = np.linalg.norm(eye_line)
    nose_to_eye_line_norm = np.linalg.norm(nose_to_eye_line)
    eye_line_unit = eye_line / eye_line_norm if eye_line_norm else eye_line
    nose_to_eye_line_unit = nose_to_eye_line / \
        nose_to_eye_line_norm if nose_to_eye_line_norm else nose_to_eye_line

    dot_product = np.dot(eye_line_unit, nose_to_eye_line_unit)
    angle = np.arccos(dot_product) * (180.0 / np.pi)

    return abs(angle - 90) < 20


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_positions = [(landmark.x, landmark.y, landmark.z)
                                  for landmark in face_landmarks.landmark]
            draw_landmarks(image, face_landmarks.landmark)
            looking_straight = compute_orientation(
                face_landmarks.landmark, image.shape)
            print("Looking straight:", looking_straight)
            print("Landmarks:", landmark_positions)

    cv2.imshow('MediaPipe Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
