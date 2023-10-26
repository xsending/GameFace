from math import hypot
import cv2
import numpy as np
from scipy.spatial import distance as dis

"""--------------------------- GLOBAL VARIABLES ----------------------------"""
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

""" -------------------------- DISTANCE COMPUTATION FUNCTIONS -------------------------- """


def calculate_face_proximity(box, image, proxy_threshold=325):
    # Get the height and width of the face bounding box
    face_width = box[2] - box[0]
    face_height = box[3] - box[1]

    # Draw rectangle to guide the user
    # Calculate the angle of diagonal using face width, height
    theta = np.arctan(face_height / face_width)

    # Use the angle to calculate height, width of the guide rectangle
    guide_height = np.sin(theta) * proxy_threshold
    guide_width = np.cos(theta) * proxy_threshold

    # Calculate the mid-point of the guide rectangle/face bounding box
    mid_x, mid_y = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2

    # Calculate the coordinates of top-left and bottom-right corners
    guide_topleft = int(mid_x - (guide_width / 2)), int(mid_y - (guide_height / 2))
    guide_bottomright = int(mid_x + (guide_width / 2)), int(mid_y + (guide_height / 2))

    # Draw the guide rectangle
    cv2.rectangle(image, guide_topleft, guide_bottomright, (0, 255, 255), 2)

    # Calculate the diagonal distance of the bounding box
    diagonal = hypot(face_width, face_height)

    # Return True if distance greater than the threshold
    return diagonal


def distance_calibration(frame):
    # capture frame if capture_image is ture and number_image_captured are less then 10
    # if we press 'c' on keyboard then capture_image become 'True'
    cv2.putText(frame, 'Capturing', (50, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 244, 255), 1)
    cv2.imwrite("ref_image.jpg", frame)


def focal_length(measured_distance, real_width, width_in_rf_image):
    f_length = (width_in_rf_image * measured_distance) / real_width
    return f_length


def distance_finder(f_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * f_length) / face_width_in_frame
    return distance


def get_face_width(image, distance_level):
    face_width = 0
    image_gpu = cv2.UMat(image)
    gray_image = cv2.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        face_width = w
        if distance_level < 10:
            distance_level = 10

    return face_width


def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]

    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)

    distance = dis.euclidean(point1, point2)
    return distance


def left_right_ratio(image, outputs, left, right):
    try:
        landmark = outputs.multi_face_landmarks[0]

        left = landmark.landmark[left]
        right = landmark.landmark[right]

        left_right_dis = euclidean_distance(image, left, right)

        return left_right_dis
    except TypeError:
        return 0


def get_aspect_ratio(image, outputs, top_bottom, left_right):
    try:
        landmark = outputs.multi_face_landmarks[0]

        top = landmark.landmark[top_bottom[0]]
        bottom = landmark.landmark[top_bottom[1]]

        top_bottom_dis = euclidean_distance(image, top, bottom)

        left = landmark.landmark[left_right[0]]
        right = landmark.landmark[left_right[1]]

        left_right_dis = euclidean_distance(image, left, right)

        aspect_ratio = left_right_dis / top_bottom_dis

        return aspect_ratio
    except TypeError:
        return 0


""" -------------------------- DISTANCE COMPUTATION FUNCTIONS -------------------------- """