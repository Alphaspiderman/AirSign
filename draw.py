import cv2
import mediapipe as mp

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detection
hand_det = mp.solutions.hands.Hands()

# Initialize drawing utilities
drawing_utils = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
    # Read frame from video capture
    _, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    op = hand_det.process(rgb_frame)
    hands = op.multi_hand_landmarks

    if hands:
        # Draw landmarks and connections on the frame for each detected hand
        for hand in hands:
            drawing_utils.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    # Display the output frame
    cv2.imshow("Output", frame)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release video capture and destroy window
cap.release()
cv2.destroyWindow("Output")
