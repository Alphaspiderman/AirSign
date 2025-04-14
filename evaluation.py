import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from feature_extraction import build_feature_extractor

# Load the saved extracted features and labels
features = np.load("signature_features.npy")
labels = np.load("signature_labels.npy")
label_map = np.load("label_map.npy", allow_pickle=True).item()

# Load the feature extraction model
feature_extractor = build_feature_extractor()


def extract_features(img):
    img = cv2.resize(img, (128, 128))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img)
    return features.flatten()


# Function to calculate cosine similarity


def evaluate_signature(image, username):
    user_folder = os.path.join("features", username)

    if not os.path.exists(user_folder):
        print(f"No features found for user: {username}")
        return None, 0.0
    
    user_features = []

    for file in os.listdir(user_folder):
        if file.endswith(".npy"):
            path = os.path.join(user_folder, file)
            user_features.append(np.load(path))

    if not user_features:
        print(f"No feature vectors available for user: {username}")
        return None, 0.0

    user_features = np.array(user_features)

    image_features = extract_features(image)

    similarities = cosine_similarity([image_features], user_features)

    most_similar_index = np.argmax(similarities)
    most_similar_label = labels[most_similar_index]

    user = label_map[most_similar_label]

    print(
        f"Test Image: {img_name} | Most similar signature is from user: {user} with similarity score (using Cosine Similarity): {similarities[0][most_similar_index]:.4f}"
    )


test_data_folder = "test_data"
calculate_cosine_similarity_for_folder(test_data_folder)
