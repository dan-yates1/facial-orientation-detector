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

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(
    color=(128, 0, 128), thickness=2, circle_radius=1)


def get_landmark_positions(face_landmarks, image_shape):
    h, w = image_shape[:2]
    # Indexes for left eye, right eye, nose tip, left ear, right ear, mouth left, mouth right
    landmark_idxs = [33, 263, 1, 234, 454, 61, 291]
    return np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h, face_landmarks.landmark[i].z * w) for i in landmark_idxs])


def construct_rotation_matrix(landmarks):
    # Define vectors
    eye_line = landmarks[1] - landmarks[0]
    ear_line = landmarks[3] - landmarks[4]
    mouth_line = landmarks[5] - landmarks[6]
    nose_vector = landmarks[2] - (landmarks[0] + landmarks[1]) / 2

    # Normalizing vectors
    eye_line = eye_line / np.linalg.norm(eye_line)
    ear_line = ear_line / np.linalg.norm(ear_line)
    mouth_line = mouth_line / np.linalg.norm(mouth_line)
    nose_vector = nose_vector / np.linalg.norm(nose_vector)

    x_axis = ear_line  # Roll
    y_axis = np.cross(ear_line, nose_vector)
    y_axis = y_axis / np.linalg.norm(y_axis)  # Pitch
    z_axis = np.cross(x_axis, y_axis)  # Yaw

    # Combine axes to form the rotation matrix
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
    return rotation_matrix


def estimate_euler_angles(rotation_matrix):
    # Convert rotation matrix to Euler angles
    euler_angles = transforms3d.euler.mat2euler(rotation_matrix, 'sxyz')
    return euler_angles


def is_looking_straight(euler_angles_deg):
    # Adjust for camera offset
    pitch_offset = -43.21761588418377
    yaw_offset = 3.6571934381692266
    roll_offset = 179.63434348738102

    adjusted_roll = euler_angles_deg[2] - \
        360 if euler_angles_deg[2] > 180 else euler_angles_deg[2]

    adjusted_pitch = euler_angles_deg[0] - pitch_offset
    adjusted_yaw = euler_angles_deg[1] - yaw_offset
    adjusted_roll = adjusted_roll - roll_offset

    if adjusted_roll < -180:
        adjusted_roll += 360
    elif adjusted_roll > 180:
        adjusted_roll -= 360

    pitch_threshold = 20
    yaw_threshold = 20
    roll_threshold = 20

    is_straight = (
        abs(adjusted_pitch) < pitch_threshold and
        abs(adjusted_yaw) < yaw_threshold and
        abs(adjusted_roll) < roll_threshold
    )
    return is_straight


# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        break

    # Flip for selfie view
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = get_landmark_positions(face_landmarks, image.shape)
            rotation_matrix = construct_rotation_matrix(landmarks)
            euler_angles = estimate_euler_angles(rotation_matrix)
            euler_angles_deg = np.degrees(euler_angles)  # Convert to degrees

            looking_straight = is_looking_straight(euler_angles_deg)

            cv2.putText(image, "pitch: " + str(np.round(
                euler_angles_deg[0], 2)), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, "yaw: " + str(np.round(
                euler_angles_deg[1], 2)), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, "roll: " + str(np.round(
                euler_angles_deg[2], 2)), (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.putText(image,
                        'Looking straight: {}'.format(looking_straight), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_looking_straight else (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Face Orientation', image)

        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=drawing_spec,
                                  connection_drawing_spec=drawing_spec)

    # Display the image
    cv2.imshow('Face Orientation', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
