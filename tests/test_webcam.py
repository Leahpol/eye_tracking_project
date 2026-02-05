import cv2 as cv
import numpy as np

#initializing the webcam (1 is my computer)
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break
    # Display the frame
    cv.imshow('frame', frame)
    # press 'q' to quit the frame
    if cv.waitKey(1) == ord('q'):
        break

# clean up after end of stream
cap.release()
cv.destroyAllWindows()