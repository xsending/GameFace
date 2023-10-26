import util_functions as ut_f
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class HaarFaceDetect:
    def __init__(self, frame):
        self.frame = frame
        self.face_width = 0
        self.text = ""

    """ -------------------------------public functions------------------------------- """
    def get_current_face_proximity(self, face, proximity_threshold, calibrate=False):
        if len(face) >= 1:
            (x, y, w, h) = face[0]
            self.face_width = w
            prox = ut_f.calculate_face_proximity((x, y, x + w, y + h), self.frame, proximity_threshold)
            if calibrate is True:
                proximity_threshold = prox * 1.2

            return prox, proximity_threshold, x, y, w, h

    def get_face_distance(self, focal_length, known_width):
        if self.face_width != 0:
            distance = ut_f.distance_finder(focal_length, known_width, self.face_width)
            distance = round(distance, 2)

            return distance

    def get_detected_face(self):
        frame_gpu = cv2.UMat(self.frame)
        gray = cv2.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=15, minSize=(100, 100),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        return face

    def get_frame(self):
        return self.frame
    """ -------------------------------public functions------------------------------- """
