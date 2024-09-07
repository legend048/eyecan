import cv2
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
# import matplotlib.pyplot as plt

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
# import random
# import pandas as pd
# import os

import pyautogui as pa
from main_get_nose import process_img

# import tkinter as tk

pa.FAILSAFE = False

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv2.VideoCapture(0)


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=True,
                                    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

def check_blink():
    ret, frame = cap.read()
    original = frame

    height, width, _ = frame.shape
    try:
      image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      detection_result = detector.detect(image)

      return {"left":detection_result.face_blendshapes[0][9].score,"right":detection_result.face_blendshapes[0][10].score}

    except:
       print("Cant find face !!")


def cal_mean(coordinates_list):
    if not coordinates_list:
        return None

    x_coords = [coord[0] for coord in coordinates_list if coord is not None]
    y_coords = [coord[1] for coord in coordinates_list if coord is not None]

    if len(x_coords) == 0 or len(y_coords) == 0:
        return None

    mean_x = sum(x_coords) / len(x_coords)
    mean_y = sum(y_coords) / len(y_coords)

    return (mean_x, mean_y)

def get_value(cap, number_of_times=100):
    values = []
    for i in range(number_of_times):
        ret, frame = cap.read()
        if not ret:
            break

        nose_coord = process_face(frame)
        if nose_coord is not None:
            values.append(nose_coord)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    mean_value = cal_mean(values)

    return mean_value

def startup(width, height):
    setup_screen = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(setup_screen, "starting setup...", (width//2 - 300, height//2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("setup", setup_screen)   
    cv2.waitKey(2000)


    setup_screen = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(setup_screen, "follow the circles", (width//2 - 300, height//2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    cv2.putText(setup_screen, "follow the circles", (width//2 - 300, height//2 - 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    radius = 10

    ## left
    cv2.circle(setup_screen, (radius + 10, height//2), radius, (255,255,255), -1)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    left = get_value(cap)
    cv2.circle(setup_screen, (radius + 10, height//2), radius, (0,0,0), -1)
    
    ## right
    cv2.circle(setup_screen, (width - radius - 10, height//2), radius, (255,255,255), -1)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    right = get_value(cap)
    cv2.circle(setup_screen, (width - radius - 10, height//2), radius, (0,0,0), -1)


    ## up
    cv2.circle(setup_screen, (width//2 , radius + 10), radius, (255,255,255), -1)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    up = get_value(cap)
    cv2.circle(setup_screen, (width//2 , radius + 10), radius, (0,0,0), -1)


    ## down
    cv2.circle(setup_screen, (width//2, height - radius - 10), radius, (255,255,255), -1)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    down = get_value(cap)
    cv2.circle(setup_screen, (width//2, height - radius - 10), radius, (0,0,0), -1)

    cv2.putText(setup_screen, "setup Complete!", (width//2 - 300, height//2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    cv2.destroyWindow("setup")

    return [int(left[0]), int(left[0])], [int(right[0]),int(right[1])], [int(up[0]), int(up[1])], [int(down[0]), int(down[1])]

def process_face(frame):
    # with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = face_detection.process(frame_rgb)
    #     height, width, _ = frame.shape
    #     if results.detections:
    #         for detection in results.detections:
    #             mp_drawing.draw_detection(frame, detection)
    #             keypoints = detection.location_data.relative_keypoints
    #             nose_tip = keypoints[2]  # Nose tip index
    #             nose_x = float(nose_tip.x * width)
    #             nose_y = float(nose_tip.y * height)
    #     return (int(nose_x), int(nose_y))
    return process_img(frame)


def calculate_centroid(coordinates):
    if not coordinates:
        raise ValueError("The list of coordinates is empty.")
    
    # Separate the x and y coordinates
    x_coords, y_coords = zip(*coordinates)
    
    # Calculate the mean of x and y coordinates
    mean_x = sum(x_coords) / len(x_coords)
    mean_y = sum(y_coords) / len(y_coords)
    return [int(mean_x), int(mean_y)]

def smooth_move(x_start, y_start, x_end, y_end, steps, duration):
    x_step = (x_end - x_start) / steps
    y_step = (y_end - y_start) / steps
    for i in range(steps):
        pa.moveTo(x_start + x_step * i, y_start + y_step * i)
        time.sleep(duration / steps)

def map_value(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

# def move_cursor(x, y):
#     root.geometry(f'+{x}+{y}')


if __name__=="__main__":

    left, right, up, down = startup(1920, 1080)

    print("\n\n",left, right, up, down,"\n\n")

    cap = cv2.VideoCapture(0)

    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)

    mean_coord = []
    per_iteration_mean = 2

    # ini_position = [0,0]

    ##-------------------------------------------------------------
    # root = tk.Tk()
    # root.overrideredirect(True)  # Removes window borders, making it an overlay
    # root.wm_attributes("-topmost", True)  # Keeps the window always on top
    # root.wm_attributes("-transparentcolor", "blue")  # Make "blue" color transparent
    # cursor_size = 20
    # canvas = tk.Canvas(root, width=cursor_size, height=cursor_size, highlightthickness=0, bg='blue')
    # canvas.pack()

    # # Draw a white circle to represent the cursor
    # cursor = canvas.create_oval(0, 0, cursor_size, cursor_size, fill="white", outline="")

    # # Set initial position
    # move_cursor(500, 500)


    ##-------------------------------------------------------------


    while True:
        ret, frame = cap.read()
        original = frame
        if not ret:
            break

        height, width, _ = frame.shape

    ## -------------------------------------------------------------------------
    ## click
        # blink_hold = 0.5 ## in seconds
        # blink_score = check_blink()
        # if blink_score["left"] >=0.7 or blink_score["right"] >=0.7:
        #     initial_time = time.time()
        #     while True:
        #         blink_score = check_blink()
        #         if blink_score["left"] >=0.7 or blink_score["right"] >=0.7:
        #             print("\n\n\nclick...\n\n\n")
        #             break
        #         new_time = time.time()
        #         if (new_time - initial_time) > blink_hold:
        #             break
    ##--------------------------------------------------------------------------

        nose_coord = process_face(frame)
        print(nose_coord)
        if nose_coord:
            x_coord = int(map_value(nose_coord[0], left[0], right[0], 10, 1900))
            y_coord = int(map_value(nose_coord[1], up[1], down[1], 10, 1050))
            
            if len(mean_coord)<=per_iteration_mean:
                mean_coord.append([x_coord,y_coord])
            
            else:
                mean_coord = []
                mean_coord.append([x_coord, y_coord])
            if len(mean_coord)==per_iteration_mean:
                position = calculate_centroid(mean_coord)
                # position[0] = round(clamp(position[0], 10, 1900))
                # position[1] = round(clamp(position[1], 10, 1050))
                # print(position)
                # cv2.rectangle(blank, (position[0],position[1]), (position[0] + 10, position[1] + 10), (255, 255, 255), -1)
                # cv2.circle(original,(position[0], position[1]), 1, (255,0,0),1)
                # cv2.imshow("original", original)
                # cv2.imshow("screen", blank)
                pa.moveTo(position[0],position[1], duration=0.2, tween=pa.easeInOutQuad)
                # cv2.rectangle(blank, (position[0],position[1]), (position[0] + 10, position[1] + 10), (0, 0, 0), -1)
                
                # smooth_move(ini_position[0], ini_position[1], position[0], position[1], 30, 0.2)

                # root.after(0.2, lambda: move_cursor(position[0], position[1]))
                # root.mainloop()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

