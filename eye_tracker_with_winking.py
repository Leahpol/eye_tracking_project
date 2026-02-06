import mediapipe as mp
import cv2 as cv
import numpy as np
import time

class EyeTrackerWithWinking:

        #constants
        RED = (48, 48, 255)
        GREEN = (48, 255, 48)
        THICKNESS = 1
        LEFT_EYE_INDICES =  [362, 385, 387, 263, 373, 380]
        RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        EYES = LEFT_EYE_INDICES + RIGHT_EYE_INDICES
        

        def __init__(self, threshold = 0.18):
                # Initialize MediaPipe Face Mesh
                # Set EAR threshold

                # initialize mediapipe face mesh
                self.mp_face_mesh = mp.solutions.face_mesh
                # face detection
                self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode = False, 
                        max_num_faces = 1, 
                        refine_landmarks = True, 
                        min_detection_confidence = 0.5,
                        min_tracking_confidence = 0.5)
                # EAR threshold
                self.threshold = threshold
                self.left_EAR = 0.0
                self.right_EAR = 0.0
                # initial state of the eyes is unknown
                self.eyes_state = ""
                # counter for blinking
                self.blink_counter = 0
                # blinking flag
                self.blink = False
                # start time of the the current minute
                self.minute_start_time = time.time()
                # minute counter since the start of the program (rounded up)
                self.minutes_count = 1
                # the duration of the longest blink
                self.blink_time = 0.0
                # the start time of the blink
                self.start_blink_time = time.time()
        
        def calculate_ear(self, eye_landmarks):
                # Compute Eye Aspect Ratio
                # Return float value

                # when face is not detected
                if len(eye_landmarks) != 12:
                        return 0.0
                L1 = eye_landmarks[0]
                L2 = eye_landmarks[1]
                L3 = eye_landmarks[2]
                L4 = eye_landmarks[3]
                L5 = eye_landmarks[4]
                L6 = eye_landmarks[5]
                self.left_EAR = (self.euclidean_distance(L2, L6) + self.euclidean_distance(L3, L5)) / (2.0 * self.euclidean_distance(L1, L4))
                R1 = eye_landmarks[6]
                R2 = eye_landmarks[7]
                R3 = eye_landmarks[8]
                R4 = eye_landmarks[9]
                R5 = eye_landmarks[10]
                R6 = eye_landmarks[11]
                self.right_EAR = (self.euclidean_distance(R2, R6) + self.euclidean_distance(R3, R5)) / (2.0 * self.euclidean_distance(R1, R4))
                avarage_EAR = (self.left_EAR + self.right_EAR) / 2.0
                return avarage_EAR
                
        def get_eye_landmarks(self, landmarks, indices, frame_w, frame_h):
                # Extract specific eye landmarks
                # Return list of (x, y) coordinates

                list_of_coordinates = []
                if landmarks:
                        face = landmarks[0]
                        eyes_landmarks = []
                        # go over indices eyes and add the landmark of that face in that index to eyes_landmarks
                        for index in indices:
                                eyes_landmarks.append(face.landmark[index])
                        # for each eye's landmark, normalize it to pixel
                        # then add it to the list of coordinates
                        for lm in eyes_landmarks:
                                pixel_x = lm.x * frame_w
                                pixel_y = lm.y * frame_h
                                if pixel_x is not None and pixel_y is not None:
                                        list_of_coordinates.append((pixel_x, pixel_y))
                return list_of_coordinates
                
        def process_frame(self, frame):
                # Process single frame
                # Return annotated frame

                # get frame dimensions for normalization
                height, width, _ = frame.shape
                # Convert the BGR image to RGB before processing
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                eye_landmarks = self.get_eye_landmarks(results.multi_face_landmarks, self.EYES, width, height)
                avarage_EAR = self.calculate_ear(eye_landmarks)
                if not self.is_winking(self.left_EAR, self.right_EAR, self.threshold, eye_landmarks, frame):
                        self.eye_state_classifier_both_eyes(avarage_EAR, self.threshold, eye_landmarks, frame)
                        self.is_blinking()
                return frame
                
        def run(self):
                # Main loop: capture, process, display
                # Handle keyboard input

                # initializing the webcam (1 is my computer)
                cap = cv.VideoCapture(1)
                if not cap.isOpened():
                        print("Cannot open camera")
                        exit()
                self.minutes()
                while cap.isOpened():
                        # Capture frame-by-frame
                        success, frame = cap.read()
                        if not success:
                                print("Failed to read frame")
                                continue
                        annotated_frame = self.process_frame(frame)
                        # flip frame so it will mirror the user
                        mirror_frame = cv.flip(annotated_frame, 1)
                        cv.imshow("Eyes Tracker", mirror_frame)
                        # press 'q' to quit the frame
                        if cv.waitKey(1) == ord('q'):
                                break
                # prints the bibking frequency to the consol at the end of the video
                avarage_blinking_frequency = (int)(self.blink_counter / (self.minutes_count))
                print(f'Blinking frequency: {avarage_blinking_frequency} blinks per minute')
                # print the longest blink time to the console at the end of the video
                print(f'Longest blink duration measurement: {self.blink_time} seconds')
                # clean up after end of stream
                cap.release()
                cv.destroyAllWindows()

        def euclidean_distance(self, point1, point2):
                # calculating and returning the euclidean distance between two points

                x1, y1 = point1
                x2, y2 = point2
                distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                return distance
        
        def eye_state_classifier_both_eyes(self, EAR, threshold, eye_landmarks, frame):
                # Classifie eye state based on EAR and annotate frame, 
                # Output eye state to console

                # when no face is detected
                if EAR == 0.0: 
                        self.eyes_state = ""
                        return
                elif EAR < threshold:
                        cv.drawContours(frame, [np.array(eye_landmarks[:6], dtype=np.int32)], -1, self.RED, self.THICKNESS)
                        cv.drawContours(frame, [np.array(eye_landmarks[6:], dtype=np.int32)], -1, self.RED, self.THICKNESS)
                        if self.eyes_state != "CLOSED":
                                self.start_blink_time = time.time()
                                self.eyes_state = "CLOSED"
                                print("The eyes are: " + self.eyes_state)
                else:
                        cv.drawContours(frame, [np.array(eye_landmarks[:6], dtype=np.int32)], -1, self.GREEN, self.THICKNESS)
                        cv.drawContours(frame, [np.array(eye_landmarks[6:], dtype=np.int32)], -1, self.GREEN, self.THICKNESS)
                        if self.eyes_state != "OPEN":
                                self.eyes_state = "OPEN"
                                print("The eyes are: " + self.eyes_state)
        
        def is_winking(self, left_EAR, right_EAR, threshold, eye_landmarks, frame):
                # checks if the person is winking

                #no face detected
                if left_EAR == 0.0 or right_EAR == 0.0:
                        return False
                if left_EAR < threshold and not right_EAR < threshold and self.eyes_state != "CLOSED":
                        cv.drawContours(frame, [np.array(eye_landmarks[:6], dtype=np.int32)], -1, self.RED, self.THICKNESS)
                        cv.drawContours(frame, [np.array(eye_landmarks[6:], dtype=np.int32)], -1, self.GREEN, self.THICKNESS)
                        if self.eyes_state != "WINKING with left eye":
                                self.eyes_state = "WINKING with left eye"
                                print(self.eyes_state)
                        return True
                elif right_EAR < threshold and not left_EAR < threshold and self.eyes_state != "CLOSED":
                        cv.drawContours(frame, [np.array(eye_landmarks[:6], dtype=np.int32)], -1, self.GREEN, self.THICKNESS)
                        cv.drawContours(frame, [np.array(eye_landmarks[6:], dtype=np.int32)], -1, self.RED, self.THICKNESS)
                        if self.eyes_state != "WINKING with right eye":
                                self.eyes_state = "WINKING with right eye"
                                print(self.eyes_state)
                        return True
                return False

        def is_blinking(self):
                # checks if the person blinked or just closed it's eyes

                if self.eyes_state == "OPEN":
                        if self.blink:
                                duration = time.time() - self.start_blink_time
                                if duration > self.blink_time:
                                        self.blink_time = duration
                                self.blink_counter += 1
                        self.blink = False
                elif self.eyes_state == "CLOSED":
                        self.blink = True
                else:
                        self.blink = False

        def minutes(self):
                # calculates the minutes passed since the start of the program 
                # updates the minutes counter

                current_time = time.time()
                if current_time - self.minute_start_time >= 60:
                        self.minutes_count += 1
                        self.minute_start_time = current_time

if __name__ == "__main__":
        tracker = EyeTrackerWithWinking()
        tracker.run()
       

        
                        



