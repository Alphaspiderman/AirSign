import cv2
import os
import glob
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Build feature extraction model


def build_feature_extractor():
    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=(128, 128, 3))

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)

    return model

# Load data for feature extraction


def load_data():
    x, y = [], []
    label_map = {}

    users = [u for u in os.listdir('signatures') if os.path.isdir(
        os.path.join('signatures', u)) and not u.startswith('.')]

    for i, user in enumerate(users):
        user_path = os.path.join('signatures', user)
        if not os.path.isdir(user_path):
            continue

        label_map[i] = user

        for img_path in glob.glob(os.path.join(user_path, 'signature_*.png')):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            img = preprocess_input(img)
            x.append(img)
            y.append(i)

    x = np.array(x)
    y = np.array(y)

    return x, y, label_map

# Extract features using the VGG16 model


def extract_features(model, x):
    features = model.predict(x)
    # Flatten the features into a 1D vector per image
    features = features.reshape(features.shape[0], -1)
    return features


def save_features():
    x, y, label_map = load_data()

    feature_extractor = build_feature_extractor()

    features = extract_features(feature_extractor, x)

    np.save('signature_features.npy', features)
    np.save('signature_labels.npy', y)
    np.save('label_map.npy', label_map)

    print(f"Extracted features shape: {features.shape}")
    print(f"Saved features and labels to disk.")


save_features()
