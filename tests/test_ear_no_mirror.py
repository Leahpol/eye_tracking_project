import cv2 as cv
import mediapipe as mp



def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance
  
WHITE = (255, 255, 255)
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

    # only if faces were detected
    if faces:
        face = faces[0]
        eyes_landmarks = []
        list_of_coordinates = []
        # go over indices eyes and add the landmark of that face in that index to eyes_landmarks
        for index in EYES:
            eyes_landmarks.append(face.landmark[index])
        # for each eye's landmark, normalize it 
        # then add it to the list of coordinates
        for lm in eyes_landmarks:
            pixel_x = lm.x * width
            pixel_y = lm.y * height
            if pixel_x is not None and pixel_y is not None:
                list_of_coordinates.append((pixel_x, pixel_y))
        L1 = list_of_coordinates[0]
        L2 = list_of_coordinates[1]
        L3 = list_of_coordinates[2]
        L4 = list_of_coordinates[3]
        L5 = list_of_coordinates[4]
        L6 = list_of_coordinates[5]
        left_EAR = (euclidean_distance(L2, L6) + euclidean_distance(L3, L5)) / (2.0 * euclidean_distance(L1, L4))
        R1 = list_of_coordinates[6]
        R2 = list_of_coordinates[7]
        R3 = list_of_coordinates[8]
        R4 = list_of_coordinates[9]
        R5 = list_of_coordinates[10]
        R6 = list_of_coordinates[11]
        right_EAR = (euclidean_distance(R2, R6) + euclidean_distance(R3, R5)) / (2.0 * euclidean_distance(R1, R4))
        avarage_EAR = (left_EAR + right_EAR) / 2.0
        text = f"EAR: {avarage_EAR}"
        cv.putText(frame, text,((int)(L6[0] + 20), (int)(L6[1] + 20)) , cv.FONT_HERSHEY_PLAIN, 1, WHITE, 2, cv.LINE_AA)
            

    # flip frame so it will mirror the user
    #mirror_frame = cv.flip(frame, 1)    
    cv.imshow('Face Mesh', frame)

    # press 'q' to quit the frame
    if cv.waitKey(1) == ord('q'):
        break
    
# clean up after end of stream
cap.release()
cv.destroyAllWindows()



