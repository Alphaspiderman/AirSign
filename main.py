import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import hog
import glob
import mediapipe as mp
from feature_extraction import save_features
#from evaluation import evaluate_signature

# Mediapipe hand detection setup
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(
    static_image_mode=False,  # Set to True if hand isn't moving much
    max_num_hands=1,  # Reduce to 1 hand to prevent false detections
    min_detection_confidence=0.8,  # Increase threshold to detect hand more confidently
    min_tracking_confidence=0.8,  # Ensure stable tracking
) 

# Global canvas for login
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Save signature
def save_signature(canvas, username, count):
    folder = os.path.join("signatures", username)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"signature_{count}.png")
    cv2.imwrite(filename, canvas)


# Signature registration using hand gesture
def capture_signature(username):
    count = 0
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    prev_x, prev_y = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )
                index_tip = hand.landmark[8]
                x = int(index_tip.x * frame_width)
                y = int(index_tip.y * frame_height)

                if 150 < x < 500 and 100 < y < 400:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 4)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        # Draw ROI box
        cv2.rectangle(frame, (150, 100), (500, 400), (255, 0, 0), 2)
        output = cv2.add(frame, canvas)
        cv2.imshow("Register", output)

        key = cv2.waitKey(1)
        if key == ord('s'):
            save_signature(canvas, username, count + 1)
            print(f"Saved signature {count + 1}")
            count += 1
            canvas[:] = 0
        elif key == ord('q'):
            break
        
        if key == ord("c"):
            print("Canvas cleared")
            canvas[:] = 0  # Properly clear the canvas instead of reassigning
        if count == 5:
            print(f"Collected 5 signatures for {username}.")
            save_features(username)
            break

    #cap.release()
    cv2.destroyAllWindows()

# Feature extractor
def extract_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    features, _ = hog(
        image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True
    )
    return features


# Authentication
def authenticate(user):
    count = 0
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    prev_x, prev_y = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )
                index_tip = hand.landmark[8]
                x = int(index_tip.x * frame_width)
                y = int(index_tip.y * frame_height)

                if 150 < x < 500 and 100 < y < 400:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 4)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        # Draw ROI box
        cv2.rectangle(frame, (150, 100), (500, 400), (255, 0, 0), 2)
        output = cv2.add(frame, canvas)
        cv2.putText(output, "Draw and press 's' to submit or 'c' to clear", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Login", output)

        key = cv2.waitKey(1)
        if key == ord('s'):
            break
        elif key == ord('q'):
            # cap.release()
            cv2.destroyWindow("Login")
            return
        if key == ord('c'):
            print("Canvas cleared")
            canvas[:] = 0

    cv2.destroyWindow("Login")

    # Evaluate signature
    #matched_user, avg_score = evaluate_signature(canvas, user)

    #if avg_score > 0.5:
        #messagebox.showinfo("Login Success", f"{user} logged in successfully!")
    #else:
       # messagebox.showerror("Login Failed", "Signature did not match.")


# GUI Handlers
def register_user():
    name = simpledialog.askstring("Register", "Enter your name:")
    if name:
        name = name.strip().replace(" ", "_")
        capture_signature(name)


def login_user():
    if not os.path.exists("signatures"):
        messagebox.showerror("Error", "No registered users found.")
        return

    users = [u for u in os.listdir("signatures") if os.path.isdir(f"signatures/{u}")]
    if not users:
        messagebox.showerror("Error", "No registered users found.")
        return

    win = tk.Toplevel(root)
    win.title("Login")
    win.geometry("400x300")
    tk.Label(win, text="Select user:").pack(pady=10)

    selected_user = tk.StringVar(win)
    selected_user.set(users[0])
    tk.OptionMenu(win, selected_user, *users).pack(pady=5)

    def proceed():
        win.destroy()
        authenticate(selected_user.get())

    tk.Button(win, text="Login", command=proceed).pack(pady=10)


def quit_app():
    root.destroy()


# Initialise Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    quit()

# Tkinter GUI
root = tk.Tk()
root.title("AirSign")
root.geometry("400x300")

tk.Label(root, text="AirSign\nSignature Authentication System", font=("Helvetica", 16)).pack(pady=20)

tk.Button(root, text="Register", command=register_user, width=20, height=2).pack(pady=10)
tk.Button(root, text="Login", command=login_user, width=20, height=2).pack(pady=10)
tk.Button(root, text="Quit", command=quit_app, width=20, height=2, bg="red", fg="white").pack(pady=20)

root.mainloop()
