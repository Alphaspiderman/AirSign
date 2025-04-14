# hand_tracking.py
import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )

    def process(self, frame, canvas, prev_x, prev_y, draw_box=True):
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(rgb_frame)

        new_x, new_y = None, None

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                index_tip = hand.landmark[8]
                x = int(index_tip.x * frame_width)
                y = int(index_tip.y * frame_height)

                if 150 < x < 500 and 100 < y < 400:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 4)
                    new_x, new_y = x, y
        else:
            new_x, new_y = None, None

        if draw_box:
            cv2.rectangle(frame, (150, 100), (500, 400), (255, 0, 0), 2)

        output = cv2.add(frame, canvas)
        return output, new_x, new_y
