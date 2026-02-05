import cv2 as cv
import mediapipe as mp

# color & thickness of the face mesh 
WHITE = (255, 255, 255)
TESSELATION_THICKNESS = 1

# initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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
    
    # Convert the BGR image to RGB before processing
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # list of landmarks of the detected faces
    faces = results.multi_face_landmarks

    # only if faces were detected
    if faces:
        face = faces[0]
        #draw the face mesh of all the landmarks of the face
        mp_drawing.draw_landmarks(
            image = frame,
            landmark_list = face,
            connections = mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles.DrawingSpec(color = WHITE, thickness = TESSELATION_THICKNESS))
   
    # flip frame so it will mirror the user
    mirror_frame = cv.flip(frame, 1)
    cv.imshow('Face Mesh', mirror_frame)

    # press 'q' to quit the frame
    if cv.waitKey(1) == ord('q'):
        break
    
# clean up after end of stream
cap.release()
cv.destroyAllWindows()