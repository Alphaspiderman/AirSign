import random
import cv2
import mediapipe as mp
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detection
hand_det = mp.solutions.hands.Hands()

# Initialize drawing utilities
drawing_utils = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define an empty canvas
canvas = None

# Keyboard controls
print("Keyboard Controls:")
print("Press 'q' to quit")
print("Press 's' to save the canvas")
print("Press 'c' to clear the canvas")

while True:
    # Read frame from video capture
    _, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Create a black canvas if not already created
    if canvas is None:
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    op = hand_det.process(rgb_frame)
    hands = op.multi_hand_landmarks

    if hands:
        # Loop through detected hands
        for hand in hands:
            # Draw landmarks and connections on the frame
            drawing_utils.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
            # Draw the index finger tip on the canvas
            index_finger_tip = hand.landmark[8]
            x = int(index_finger_tip.x * frame_width)
            y = int(index_finger_tip.y * frame_height)
            cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)

    # Overlay the canvas on the frame
    frame = cv2.add(frame, canvas)
    # Display the output frame
    cv2.imshow("Output", frame)
    cv2.imshow("Canvas", canvas)

    # Keyboard controls and delay between getting next frame
    key = cv2.waitKey(1)
    if key == ord("q"):
        print("Quitting")
        break
    if key == ord("s"):
        print("Canvas saved as output.png")
        cv2.imwrite("output.png", canvas)
    if key == ord("c"):
        print("Canvas cleared")
        canvas = None

# Release video capture and destroy window
cap.release()
cv2.destroyWindow("Output")
cv2.destroyWindow("Canvas")
