import util_functions as ut_f
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9)


def get_face_angles(facial_landmarks, focal_length, frame_w, frame_h, face_2d, face_3d):
    for face_landmarks in facial_landmarks.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * frame_w, lm.y * frame_h)
                    nose_3d = (lm.x * frame_w, lm.y * frame_h, lm.z * 3000)

                x, y = int(lm.x * frame_w), int(lm.y * frame_h)

                # Get 2d coordinates
                face_2d.append([x, y])

                # Get 3d coordinates
                face_3d.append([x, y, lm.z])

        # Convert to numpy array
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert to numpy array
        face_3d = np.array(face_3d, dtype=np.float64)

        # Cam matrix

        cam_matrix = np.array([[focal_length, 0, frame_h / 2],
                               [0, focal_length, frame_w / 2],
                               [0, 0, 1]])

        # distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve pnp
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # get y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        return x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix, nose_2d, nose_3d


class MpFaceDetect:
    def __init__(self, frame):
        self.frame = frame

    def get_face_landmarks(self):
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        facial_landmarks = face_mesh.process(frame)

        return facial_landmarks

    def get_left_right_lips_distance(self, ref_left_right_lips, facial_landmarks, distance):
        left_right_lips_distance = (ut_f.left_right_ratio(self.frame, facial_landmarks, 78, 308) * distance) / ref_left_right_lips

        return left_right_lips_distance

    def get_ratio_lips(self, facial_landmarks, upper_lower_lips_coords, left_right_lips_coords):
        ratio_lips = ut_f.get_aspect_ratio(self.frame, facial_landmarks, upper_lower_lips_coords, left_right_lips_coords)

        return ratio_lips

    def draw_landmarks(self, facial_landmarks):
        for face_landmarks in facial_landmarks.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=self.frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    def get_frame(self):
        return self.frame

