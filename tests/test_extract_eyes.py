import cv2 as cv
import mediapipe as mp
import numpy as np

# color & thickness of eyes overlay
RED = (48, 48, 255)
GREEN = (48, 255, 48)
THICKNESS = 1

LEFT_EYE_INDICES =  [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
EYES = LEFT_EYE_INDICES + RIGHT_EYE_INDICES




# initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh

# face detection
face_mesh = mp_face_mesh.FaceMesh(static_image_mode = False, 
                                  max_num_faces = 1, 
                                  refine_landmarks = True, 
                                  min_detection_confidence = 0.5,
                                  min_tracking_confidence = 0.5)

# initializing the webcam (1 is my computer)
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        continue

    # get frame dimensions for normalization
    height, width, _ = frame.shape
    # Convert the BGR image to RGB before processing
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # list of landmarks of the detected faces
    faces = results.multi_face_landmarks
   
   #only if faces were detected
    if faces:
        face = faces[0]
        eyes_landmarks = []
        list_of_coordinates = []
        # go over indices eyes and add the landmark of that face in that index to eyes_landmarks
        for index in EYES:
            eyes_landmarks.append(face.landmark[index])
        # for each eye's landmark, normalize it to pixel
        # then add it to the list of coordinates
        for lm in eyes_landmarks:
            pixel_x = lm.x * width
            pixel_y = lm.y * height
            list_of_coordinates.append((pixel_x, pixel_y))
        # Draw eye contour of the left eye
        cv.drawContours(frame, [np.array(list_of_coordinates[:6], dtype=np.int32)], -1, RED, THICKNESS)
        # Draw eye contour of the right eye
        cv.drawContours(frame, [np.array(list_of_coordinates[6:], dtype=np.int32)], -1, GREEN, THICKNESS)
          
    # flip frame so it will mirror the user
    mirror_frame = cv.flip(frame, 1)    
    cv.imshow('Face Mesh', mirror_frame)

    # press 'q' to quit the frame
    if cv.waitKey(1) == ord('q'):
        break
    
# clean up after end of stream
cap.release()
cv.destroyAllWindows()