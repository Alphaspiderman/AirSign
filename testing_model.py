import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from model_training import build_feature_extractor

# Load the saved extracted features and labels
features = np.load('signature_features.npy')
labels = np.load('signature_labels.npy')
label_map = np.load('label_map.npy', allow_pickle=True).item()

# Load the feature extraction model
feature_extractor = build_feature_extractor()


def extract_single_image_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img)
    return features.flatten()

# Function to calculate cosine similarity


def calculate_cosine_similarity_for_folder(test_data_folder):

    for img_name in os.listdir(test_data_folder):
        if img_name.endswith('.png') and img_name.startswith('signature_'):
            test_image_path = os.path.join(test_data_folder, img_name)

            test_image_features = extract_single_image_features(
                test_image_path)

            similarities = cosine_similarity([test_image_features], features)

            most_similar_index = np.argmax(similarities)
            most_similar_label = labels[most_similar_index]

            user = label_map[most_similar_label]

            print(
                f"Test Image: {img_name} | Most similar signature is from user: {user} with similarity score: {similarities[0][most_similar_index]:.4f}")


test_data_folder = 'test_data'
calculate_cosine_similarity_for_folder(test_data_folder)
