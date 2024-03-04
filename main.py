import cv2
import mediapipe as mp
import numpy as np

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
    nose_tip = landmarks[4]
    left_eye = landmarks[130]
    right_eye = landmarks[359]
    
    points = np.array([(landmark.x * image_shape[1], landmark.y * image_shape[0]) for landmark in [nose_tip, left_eye, right_eye]])
    
    nose_to_left_eye = points[1] - points[0]
    nose_to_right_eye = points[2] - points[0]
    
    angle = np.degrees(np.arctan2(nose_to_right_eye[1] - nose_to_left_eye[1], nose_to_right_eye[0] - nose_to_left_eye[0]))
    
    return abs(angle) < 10


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
            draw_landmarks(image, face_landmarks.landmark)
            
            looking_straight = compute_orientation(face_landmarks.landmark, image.shape)
            print(looking_straight)  # True or False based on face orientation.
            
            if not looking_straight:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.imshow('MediaPipe Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit.
        break

cap.release()
cv2.destroyAllWindows()