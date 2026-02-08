import unittest
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from eye_tracker import EyeTracker

class TestEyeTracker(unittest.TestCase):

    tracker = EyeTracker()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    eye_landmarks = [[0, 0], [8, 0], [20, 0], [30, 0], [37, 0], [50, 0], 
                     [0, 10], [10, 15], [20, 10], [30, 15], [40, 10], [50, 10]]

    def test_ear_calculation(self):

        exp_left_EAR = 0.9833
        exp_right_EAR = 0.9915
        exp_avg_EAR = 0.9874
        avg_EAR = self.tracker.calculate_ear(self.eye_landmarks)
        self.assertAlmostEqual(avg_EAR, exp_avg_EAR, places=4)
        self.assertAlmostEqual(self.tracker.left_EAR, exp_left_EAR, places=4)
        self.assertAlmostEqual(self.tracker.right_EAR, exp_right_EAR, places=4)

    def test_eye_state(self):

        ear_values  = [0.18, 0.08, 0.2, 0.0]  
        expected_eyes_states = ["OPEN", "CLOSED", "OPEN", "face not detected"]
        eyes_states = []
        for ear in ear_values:
            self.tracker.eye_state_classifier_both_eyes(ear, 0.18, self.eye_landmarks, self.frame)
            eyes_states.append(self.tracker.eyes_state)
        self.assertEqual(eyes_states, expected_eyes_states)

    def test_winking(self):

        left_ear_values  = [0.18, 0.08, 0.2, 0.2, 0.3, 0.17, 0.0]  
        right_ear_values = [0.3, 0.12, 0.2, 0.17, 0.3, 0.3, 0.2] 
        expected_eyes_states = [False, False, False, True, False, True, False]
        eyes_states = []
        for left_EAR, right_EAR in zip(left_ear_values, right_ear_values):
            eyes_states.append(self.tracker.is_winking(left_EAR, right_EAR, 0.18, self.eye_landmarks, self.frame))
        self.assertEqual(eyes_states, expected_eyes_states)

    def test_blinking(self):

        self.tracker.start_blink_time = time.time()
        self.tracker.pass_first_open = 0
        self.tracker.pass_first_closed = 0
        #normal blinking
        self.tracker.eyes_state = "OPEN"
        self.tracker.is_blinking()
        self.assertFalse(self.tracker.is_blink)
        self.tracker.eyes_state = "CLOSED"
        self.tracker.pass_first_open = 2
        self.tracker.pass_first_closed = 1
        self.tracker.is_blinking()
        self.assertTrue(self.tracker.is_blink)
        self.tracker.eyes_state = "OPEN"
        self.tracker.is_blinking()
        self.assertFalse(self.tracker.is_blink)
        self.assertEqual(self.tracker.blink_counter, 1)
        # closed eyes, face not detected, open eyes again
        self.tracker.eyes_state = "face not detected"
        self.tracker.is_blinking()
        self.assertFalse(self.tracker.is_blink)
        self.tracker.eyes_state = "CLOSED"
        self.tracker.is_blinking()
        self.tracker.eyes_state = ""
        self.tracker.is_blinking()
        self.tracker.eyes_state = "OPEN"
        self.tracker.is_blinking()
        self.assertEqual(self.tracker.blink_counter, 1)
        # closed eyes (sleeping), open eyes (wake up)
        self.tracker.pass_first_open = 0
        self.tracker.pass_first_closed = 0
        self.tracker.eyes_state = "CLOSED"
        self.tracker.is_blinking()
        self.tracker.eyes_state = "OPEN"
        self.tracker.is_blinking()
        self.assertEqual(self.tracker.blink_counter, 1)



if __name__ == "__main__":
    unittest.main()
