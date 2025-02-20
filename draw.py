import os
import time
import warnings

# Set the environment variable to suppress Tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import numpy as np
from cv2.typing import MatLike

# Ignore warnings
warnings.filterwarnings("ignore")

# Initialize video capture
cap = cv2.VideoCapture(0)


# Initialize hand detection
hand_det = mp.solutions.hands.Hands()

# Initialize drawing utilities
drawing_utils = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Get the first frame to define the canvas dimensions
_, frame = cap.read()
frame_height, frame_width, _ = frame.shape

# Define an empty canvas
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Control when to process the handtracking
frame_queue = []
real_time = False

# Wait for Tensorflow to load and clear the console
print("Waiting for Tensorflow to load")
time.sleep(1)
os.system("cls" if os.name == "nt" else "clear")

# Ask user if they want to process in real time
if input("Process in Real Time? (y/n):").lower() == "y":
    real_time = True
    print("Processing in Real Time")
    # Keyboard controls
    print("Keyboard Controls:")
    print("Press 'q' to quit")
    print("Press 's' to save the canvas")
    print("Press 'c' to clear the canvas")
else:
    print("Processing in Post Processing Mode")
    print("Press 'q' to stop recording")


def process_frame(frame: MatLike, canvas: MatLike):
    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

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

    return frame, canvas


def handle_keyboard_input(key, canvas, real_time):
    if key == ord("q"):
        print("Quitting")
        return True
    if key == ord("s"):
        print("Canvas saved as output.png")
        cv2.imwrite("output.png", canvas)
    if key == ord("c") and real_time:
        print("Canvas cleared")
        canvas = np.zeros_like(canvas)
    return False


while True:
    # Read frame from video capture
    _, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if real_time:
        # Process the frame for hand detection (only in real-time mode)
        frame, canvas = process_frame(frame, canvas)
        # Overlay the canvas on the frame
        frame = cv2.add(frame, canvas)
        # Display the canvas
        cv2.imshow("Canvas", canvas)
    else:
        # Save the frame to the queue (only in post-processing mode)
        frame_queue.append(frame)

    # Display the output frame
    cv2.imshow("Output", frame)

    # Keyboard controls and delay between getting next frame
    key = cv2.waitKey(1)
    if handle_keyboard_input(key, canvas, real_time):
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

# If not real-time, process the frames in the queue
if not real_time:
    # Clear the console
    os.system("cls" if os.name == "nt" else "clear")
    # Inform the user
    remaining = len(frame_queue)
    print(f"Processing the frames ({remaining} frames)")
    # Process the frames
    while frame_queue:
        remaining = len(frame_queue)
        if remaining % 100 == 0:
            print(f"Remaining Frames: {remaining}")
        _, canvas = process_frame(frame_queue.pop(0), canvas)
    cv2.imshow("Canvas", canvas)
    # Inform the user and wait
    print("Processing complete")
    print("Press any key to exit")
    cv2.waitKey(0)
# Save the final canvas
print("Final Canvas saved as final_output.png")
cv2.imwrite("final_output.png", canvas)
# Destroy all windows
cv2.destroyAllWindows()
