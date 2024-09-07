import mediapipe as mp
import cv2
import numpy as np

# Define the path to your model.
model_path = 'face_landmarker_v2_with_blendshapes.task'

# Load the necessary modules from MediaPipe.
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure FaceLandmarker options for image input.
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

# Function to display the face landmarks on the image.
def draw_landmarks(image, face_landmarks_list):
    for face_landmarks in face_landmarks_list:
        for landmark in face_landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

def process_img(frame):

    height, width, _ = frame.shape

    # Convert the OpenCV image to the MediaPipe Image format.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Use the Face Landmarker to detect landmarks.
    with FaceLandmarker.create_from_options(options) as landmarker:
        face_landmarker_result = landmarker.detect(mp_image)

        # Check if any faces were detected.
        if face_landmarker_result.face_landmarks:
            print(f"Detected {len(face_landmarker_result.face_landmarks)} faces.")
            
            # Print the coordinates of the first face's landmarks.
            for i, landmark in enumerate(face_landmarker_result.face_landmarks[0]):
                # print(f"Landmark #{i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")

                if i == 1: ## 1for Nose-Tip
                    # return (int(landmark.x*width), int(landmark.y*height))
                    return (landmark.x*width, landmark.y*height)
                

if __name__=="__main__":

    img = cv2.imread("smile.png")
    cv2.circle(img, process_img(img), 2, (255,0,0), 1)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()