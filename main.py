import time
import cv2
import asyncio
import numpy as np
from tkinter import ttk, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import mouse
import keyboard
import subprocess
from Segment import Segment
import HaarFaceDetect
import MpFaceDetect
import util_functions as ut_f

UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]
LEFT_RIGHT_CHEEK = [58, 288]

""" -------------------------- GLOBAL VARIABLES FOR DISTANCE COMPUTATIONS -------------------------- """
ref_left_right_lips = 0
ref_left_right_lips_distance = 0
mouth_length = 0
known_distance = 13  # Inches
ref_distance = 0
known_width = 1.2  # Inches
distance_level = 0
distance = 0
ref_image = ""
ref_face_width = None
focal_length = None
scale_factor = 0.25  # Adjust this value as desired (0.8 represents 80% of the screen)
proximity = 0
proximity_threshold = 0
prox_threshold = 0
prox = 0
""" -------------------------- GLOBAL VARIABLES FOR DISTANCE COMPUTATIONS -------------------------- """

""" -------------------------- GLOBAL VARIABLE CONTROLS -------------------------- """
string1 = 'c'
string2 = 'x'
string3 = 'z'
string4 = 'a'
string5 = 'd'
string6 = 'w'
string7 = 's'

last_key_pressed_face = ""
last_key_pressed_head = ""
last_mouse_key_pressed = ""
""" -------------------------- GLOBAL VARIABLE CONTROLS -------------------------- """

cap = cv2.VideoCapture(0)

""" -------------------------- CONDITION FLAGS -------------------------- """
captured = False
is_calibrated = False
is_mouse_mode = False
mode_flag = 0
mouse_click_left = False
mouse_click_right = False
head_key_pressed = False
single_press_mode = False
smiling = False
is_upright = False
osk_process = None
""" -------------------------- CONDITION FLAGS -------------------------- """

""" -------------------------- FONTS -------------------------- """
fonts = cv2.FONT_HERSHEY_COMPLEX
fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonts4 = cv2.FONT_HERSHEY_TRIPLEX
""" -------------------------- FONTS -------------------------- """


def open_osk():
    global osk_process
    osk_process = subprocess.Popen('osk.exe')


# Function to close the on-screen keyboard
def close_osk():
    global osk_process
    if osk_process:
        osk_process.terminate()
        osk_process = None


def single_press():
    global single_press_mode
    print("single press mode on")
    single_press_mode = True


def multi_press():
    global single_press_mode
    print("multi press mode on")
    single_press_mode = False


def re_calibrate():
    global is_calibrated
    if is_calibrated:
        is_calibrated = False
        time.sleep(1)


# Define the function to submit the entry forms
def submit_forms():
    entry_fields = []
    for child in middle_frame.winfo_children():
        if isinstance(child, Segment):
            face_movement_widget = child.winfo_children()[0]
            entry_widget = child.winfo_children()[1]  # index 0 is the label, index 1 is the entry widget
            entry_value = entry_widget.get()
            entry_fields.append(entry_value)
            entry_widget.delete(0, 'end')  # Clear the entry field value
            if entry_value != '':
                if len(entry_value) == 1:
                    label_widget = child.winfo_children()[2]
                    label_widget.configure(text=entry_value)
                else:
                    if entry_value == "left" or entry_value == "right" or entry_value == "up" or entry_value == "down":
                        label_widget = child.winfo_children()[2]
                        label_widget.configure(text=entry_value)
                    else:
                        messagebox.showerror("Error", face_movement_widget.cget("text") + " entry has an invalid input")
                        return

    print(entry_fields)
    global string1
    global string2
    global string3
    global string4
    global string5
    global string6
    global string7
    string1 = entry_fields[0] if entry_fields[0] != '' else string1
    string2 = entry_fields[1] if entry_fields[1] != '' else string2
    string3 = entry_fields[2] if entry_fields[2] != '' else string3
    string4 = entry_fields[3] if entry_fields[3] != '' else string4
    string5 = entry_fields[4] if entry_fields[4] != '' else string5
    string6 = entry_fields[5] if entry_fields[5] != '' else string6
    string7 = entry_fields[6] if entry_fields[6] != '' else string7


""" -------------------------- GUI LOGIC-------------------------- """

# Create a main window
window = ctk.CTk()
window.title('GameFace')
window.geometry('640x480+0+0')
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
# Add a Notebook widget to the window
notebook = ttk.Notebook(window)

# Create the first tab
tab1 = ctk.CTkFrame(notebook, width=500, height=500)
notebook.add(tab1, text='Tab 1')

# Add frame to first tab
label1 = ttk.Label(tab1)
label1.pack()

# Create the second tab
tab2 = ctk.CTkFrame(notebook, width=500, height=500)
notebook.add(tab2, text='Tab 2')

# Create top frame of the second tab
top_frame = ctk.CTkFrame(tab2, width=450, height=450, fg_color='#f77f00', corner_radius=0)
top_frame.columnconfigure(0, weight=1)
top_frame.rowconfigure(0, weight=1)
top_frame.pack(side='top', expand=False, fill='x')

# Add widgets to top frame
title = ctk.CTkLabel(top_frame,
                     text="KEY BINDING SETTINGS",
                     width=50, justify='center',
                     text_color='black',
                     font=('Arial Baltic', 20))
title.grid(row=0, column=0, pady=20)

# Create middle frame of the second tab
middle_frame = ctk.CTkFrame(tab2, fg_color='#003049')
middle_frame.pack(expand=True, fill='both')

# Add widgets to the bottom frame
Segment(middle_frame, 'Lean Forward', string1)
Segment(middle_frame, 'Open Mouth', string2)
Segment(middle_frame, 'Smile', string3)
Segment(middle_frame, 'Look Left', string4)
Segment(middle_frame, 'Look Right', string5)
Segment(middle_frame, 'Look Up', string6)
Segment(middle_frame, 'Look Down', string7)

# Create bottom  of the second tab
bottom_frame = ctk.CTkFrame(tab2, fg_color='#003049')
bottom_frame.rowconfigure((0, 1, 2), weight=1, uniform='a')
bottom_frame.columnconfigure((0, 1), weight=1, uniform='a')
bottom_frame.pack(expand=True, fill='both')

# Create the submit button
submit_button = ctk.CTkButton(bottom_frame, text="Submit", border_spacing=5, text_color='#000', fg_color='#f77f00', corner_radius=8, command=submit_forms)
submit_button.grid(row=0, column=0, sticky='e', padx=5)

single_press_button = ctk.CTkButton(bottom_frame, text="Single press mode", border_spacing=5, command=single_press)
single_press_button.grid(row=0, column=1, sticky='w', padx=5)

multi_press_button = ctk.CTkButton(bottom_frame, text="Multi press mode", border_spacing=5, command=multi_press)
multi_press_button.grid(row=1, column=1, sticky='w', padx=5)

calibrate_button = ctk.CTkButton(bottom_frame, text="Re-calibrate", border_spacing=5, command=re_calibrate)
calibrate_button.grid(row=1, column=0, sticky='e', padx=5)

notebook.pack(expand=True, fill='both')

""" -------------------------- GUI LOGIC-------------------------- """


async def head_distance_movement_controls(haar_object):
    global distance_level, prox, smiling, is_upright, distance, ref_left_right_lips_distance
    global last_key_pressed_head, is_mouse_mode, mode_flag, proximity, proximity_threshold

    frame = haar_object.get_frame()
    face = haar_object.get_detected_face()

    if len(face) >= 1:
        prox, proximity_threshold, x, y, w, h = haar_object.get_current_face_proximity(face, proximity_threshold)
        distance = haar_object.get_face_distance(focal_length, known_width)
        text = ""
        """ --------------------- HEAD DISTANCE MOVEMENT CONTROLS --------------------- """
        if prox > proximity_threshold:
            text = "Leaning forward"
            if last_key_pressed_head != "":
                keyboard.release(last_key_pressed_head)
            keyboard.press(string1)
            last_key_pressed_head = string1
        elif prox < proximity - (proximity_threshold - proximity + 10):
            text = "Leaning backward"
            if is_upright and mode_flag == 0:
                mode_flag = 1
                is_mouse_mode = True

        elif proximity - (proximity_threshold - proximity) <= prox < proximity_threshold:
            text = "Upright"
            if mode_flag == 1:
                mode_flag = 0
        """ --------------------- HEAD DISTANCE MOVEMENT CONTROLS --------------------- """
        cv2.putText(frame, text, (x - 6, y - 6), fonts, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        return frame
    else:
        return frame


async def facial_movement_and_head_rotation_controls(mp_object, start):
    global last_key_pressed_face, smiling, is_upright, last_key_pressed_head, head_key_pressed
    global ref_left_right_lips_distance, is_mouse_mode, screen_width, screen_height

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.

    frame = mp_object.get_frame()
    facial_landmarks = mp_object.get_face_landmarks()

    frame_h, frame_w, frame_c = frame.shape
    face_3d = []
    face_2d = []

    if facial_landmarks.multi_face_landmarks:
        left_right_lips_distance = \
            mp_object.get_left_right_lips_distance(ref_left_right_lips, facial_landmarks, distance)
        ratio_lips = mp_object.get_ratio_lips(facial_landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)

        """ --------------------- FACE MOVEMENT CONTROLS --------------------- """
        if ratio_lips < 2.5:
            text1 = "Open mouth"
            # smiling = False
            if last_key_pressed_face != "":
                keyboard.release(last_key_pressed_face)
            keyboard.press(string2)
            last_key_pressed_face = string2
        elif (15 > ratio_lips > 3.3) and left_right_lips_distance > 16:
            text1 = "Smiling"
            if last_key_pressed_face != "":
                keyboard.release(last_key_pressed_face)
            keyboard.press(string3)
            last_key_pressed_face = string3
        else:
            text1 = "Neutral"
            # smiling = False
            if last_key_pressed_face != "":
                keyboard.release(last_key_pressed_face)
        # switch to mouse mode

        """ --------------------- FACE MOVEMENT CONTROLS --------------------- """

        cv2.putText(frame, text1, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix, nose_2d, nose_3d = \
            MpFaceDetect.get_face_angles(facial_landmarks, focal_length, frame_w, frame_h, face_2d, face_3d)

        """ --------------------- HEAD MOVEMENT CONTROLS --------------------- """
        # see where head faces
        if y < -7:
            text2 = "Looking left"
            if single_press_mode:
                if last_key_pressed_head != "":
                    keyboard.release(last_key_pressed_head)
                if not head_key_pressed:
                    keyboard.press(string4)
                    head_key_pressed = True
                last_key_pressed_head = string4
            else:
                if last_key_pressed_head != "":
                    keyboard.release(last_key_pressed_head)
                keyboard.press(string4)
                last_key_pressed_head = string4
        elif y > 7:
            text2 = "Looking right"
            if single_press_mode:
                if last_key_pressed_head != "":
                    keyboard.release(last_key_pressed_head)
                if not head_key_pressed:
                    keyboard.press(string5)
                    head_key_pressed = True
                last_key_pressed_head = string5
            else:
                if last_key_pressed_head != "":
                    keyboard.release(last_key_pressed_head)
                keyboard.press(string5)
                last_key_pressed_head = string5
        elif x < -10:
            text2 = "Looking down"
            if single_press_mode:
                if last_key_pressed_head != "":
                    keyboard.release(last_key_pressed_head)
                if not head_key_pressed:
                    keyboard.press(string7)
                    head_key_pressed = True
                last_key_pressed_head = string7
            else:
                if last_key_pressed_head != "":
                    keyboard.release(last_key_pressed_head)
                keyboard.press(string7)
                last_key_pressed_head = string7

        elif x > 15:
            text2 = "Looking up"
            if single_press_mode:
                if last_key_pressed_head != "":
                    keyboard.release(last_key_pressed_head)
                if not head_key_pressed:
                    keyboard.press(string6)
                    head_key_pressed = True
                last_key_pressed_head = string6
            else:
                if last_key_pressed_head != "":
                    keyboard.release(last_key_pressed_head)
                keyboard.press(string6)
                last_key_pressed_head = string6
        else:
            text2 = "Forward"
            if last_key_pressed_head != "":
                keyboard.release(last_key_pressed_head)
            if head_key_pressed:
                head_key_pressed = False

        if text2 == "Forward":
            is_upright = True
        else:
            is_upright = False
        """ --------------------- HEAD MOVEMENT CONTROLS --------------------- """
        # display nose direction
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

        cv2.line(frame, p1, p2, (255, 0, 0), 3)

        # text on frame
        cv2.putText(frame, text2, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(frame, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        end = time.time()
        total_time = end - start
        fps = 1 / total_time
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_object.draw_landmarks(facial_landmarks)
        frame = mp_object.get_frame()

    # Flip the image horizontally for a selfie-view display.

    return frame


def mouse_mode(frame, height, width, start, mp_object, haar_object):
    global is_mouse_mode, mouse_click_left, mouse_click_right, \
        distance, mode_flag, proximity, proximity_threshold, prox_threshold, prox
    # pyautogui.FAILSAFE = False
    # Calculate the position of the bounding box

    bbox_x1 = int(width * scale_factor)
    bbox_y1 = int(height * scale_factor)
    bbox_x2 = int(width - (width * scale_factor))
    bbox_y2 = int(height - (height * scale_factor))

    # Create the bounding box
    bounding_box = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.

    facial_landmarks = mp_object.get_face_landmarks()

    face = haar_object.get_detected_face()

    left_right_lips_distance = \
        mp_object.get_left_right_lips_distance(ref_left_right_lips, facial_landmarks, distance)
    ratio_lips = mp_object.get_ratio_lips(facial_landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)

    if len(face) >= 1:
        prox, proximity_threshold, x, y, w, h = haar_object.get_current_face_proximity(face, proximity_threshold)
        distance = haar_object.get_face_distance(focal_length, known_width)
        text = ""
        """ --------------------- HEAD DISTANCE MOVEMENT CONTROLS --------------------- """
        if prox > proximity_threshold:

            text = "Leaning forward"

        elif prox < proximity - (proximity_threshold - proximity + 10):
            text = "Leaning backward"
            if mode_flag == 0:
                mode_flag = 1
                close_osk()
                is_mouse_mode = False

        elif proximity - (proximity_threshold - proximity) <= prox < proximity_threshold:
            text = "Upright"
            if mode_flag == 1:
                mode_flag = 0
        """ --------------------- HEAD DISTANCE MOVEMENT CONTROLS --------------------- """

        cv2.putText(frame, text, (x - 6, y - 6), fonts, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

    if facial_landmarks.multi_face_landmarks:
        face_landmarks = facial_landmarks.multi_face_landmarks[0]  # Assuming only one face is detected
        landmarks = face_landmarks.landmark
        for id, landmark in enumerate(landmarks[474:477]):
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if id == 1:
                cursor_x = screen_width - (
                        screen_width / (width / (scale_factor * 10)) * (x - (width * scale_factor)))
                cursor_y = screen_height / (height / (scale_factor * 10)) * (y - (height * scale_factor))

                mouse.move(cursor_x, cursor_y, absolute=True)

        if ratio_lips < 2.5:
            # if last_mouse_key_pressed != "":
            #     mouse.release(last_mouse_key_pressed)
            if not mouse_click_left:
                mouse.click("left")
                mouse_click_left = True
                mouse_click_right = False
        elif (15 > ratio_lips > 3.3) and left_right_lips_distance > 16:
            # if last_mouse_key_pressed != "":
            #     mouse.release(last_mouse_key_pressed)
            if not mouse_click_right:
                mouse.click("right")
                mouse_click_right = True
                mouse_click_left = False
        else:
            mouse_click_left = False
            mouse_click_right = False
        # switch to keyboard mode

    cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]),
                  (0, 255, 0), 2)

    if keyboard.is_pressed("down"):
        keyboard.write("down")
    elif keyboard.is_pressed("up"):
        keyboard.write("up")
    elif keyboard.is_pressed("left"):
        keyboard.write("left")
    elif keyboard.is_pressed("right"):
        keyboard.write("right")
    elif keyboard.is_pressed("esc"):
        keyboard.write("esc")

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    return frame


def calibration(frame, height, width, mp_object, haar_object):
    global is_calibrated, ref_image, ref_face_width, focal_length, ref_distance
    global ref_left_right_lips, ref_left_right_lips_distance, mouth_length
    global prox, prox_threshold, proximity, proximity_threshold

    facial_landmarks = mp_object.get_face_landmarks()

    face = haar_object.get_detected_face()

    cv2.putText(frame, f"Calibrating Distance. Position your face", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 127, 255), 2)
    cv2.putText(frame, f"center and upright in front of the camera.", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 127, 255), 2)
    cv2.putText(frame, f"Open mouth to capture image", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255), 2)
    cv2.putText(frame, f"You will not be able to proceed without calibrating.", (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 127, 255), 2)

    cv2.putText(frame, f"Height: {height}", (500, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255), 2)
    cv2.putText(frame, f"Width:  {width}", (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255), 2)
    # cv2.imshow('Combined', frame)

    if len(face) >= 1:
        prox, prox_threshold, x, y, w, h = haar_object.get_current_face_proximity(face, prox_threshold, True)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

    cv2.putText(frame, 'Proximity threshold: {:.2f}'.format(prox_threshold),
                (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255), 2)

    ref_left_right_lips = ut_f.left_right_ratio(frame, facial_landmarks, 78, 308)

    if ref_left_right_lips == 0:
        return frame

    ref_left_right_lips_distance = known_distance
    # Draw the face mesh annotations on the image.

    if facial_landmarks.multi_face_landmarks:
        ratio_lips = mp_object.get_ratio_lips(facial_landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)

        """ ---------------- CAPTURES IMAGE IN CALIBRATION TO SERVE AS REFERENCE IMAGE ---------------- """
        if ratio_lips < 1.8:
            proximity = prox
            proximity_threshold = prox_threshold
            ut_f.distance_calibration(frame)
            ref_image = cv2.imread("ref_image.jpg")
            ref_face_width = ut_f.get_face_width(ref_image, distance_level)
            if ref_face_width == 0:
                return frame

            focal_length = ut_f.focal_length(known_distance, known_width, ref_face_width)
            ref_distance = ut_f.distance_finder(focal_length, known_width, ref_face_width)
            ref_distance = round(ref_distance, 2)
            is_calibrated = True
            return frame
        else:
            return frame
            # cv2.destroyWindow("Combined")

    """ ---------------- CAPTURES IMAGE IN CALIBRATION TO SERVE AS REFERENCE IMAGE ---------------- """

    """ ALWAYS RUN ONCE APP IS OPENED FOR CALIBRATION """


async def main():
    global is_calibrated, is_mouse_mode

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.flip(frame.copy(), 1)
        start = time.time()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        haar_object = HaarFaceDetect.HaarFaceDetect(frame)
        mp_object = MpFaceDetect.MpFaceDetect(frame)
        height, width = frame.shape[0:2]
        """ ALWAYS RUN ONCE APP IS OPENED FOR CALIBRATION """
        if not is_calibrated:
            frame = calibration(frame, height, width, mp_object, haar_object)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img)
            # Update the label widget with the new frame
            label1.configure(image=img_tk)
            label1.image = img_tk

            # Update the GUI window
            window.update()

        else:
            """ ---------------- RUNS IF USER ENTERS MOUSE MODE ---------------- """
            if is_mouse_mode:
                frame = mouse_mode(frame, height, width, start, mp_object, haar_object)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(img)
                # Update the label widget with the new frame
                label1.configure(image=img_tk)
                label1.image = img_tk

                # Update the GUI window
                window.update()

                """ ---------------- RUNS IF USER ENTERS MOUSE MODE ---------------- """
            else:

                """ ---------------- DEFAULT MODE (KEYBOARD MODE) - RUNS IF USER NOT IN MOUSE MODE ---------------- """
                t1 = asyncio.create_task(head_distance_movement_controls(haar_object))
                t2 = asyncio.create_task(facial_movement_and_head_rotation_controls(mp_object, start))

                frame1 = await t1
                frame2 = await t2

                # Combine the two modified frames
                frame_combined = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

                # Convert the frame to a format that can be displayed by tkinter
                frame_rgb = cv2.cvtColor(frame_combined, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(img)

                # Update the label widget with the new frame
                label1.configure(image=img_tk)
                label1.image = img_tk

                # Update the GUI window
                window.update()

                if cv2.waitKey(1) == ord('q'):
                    break

                # Display the combined frame
                """ ---------------- DEFAULT MODE (KEYBOARD MODE) - RUNS IF USER NOT IN MOUSE MODE ---------------- """

    cap.release()

    # Start the Tkinter event loop
    window.mainloop()


asyncio.run(main())
