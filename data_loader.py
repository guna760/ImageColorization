import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_images(folder):
    X = []
    Y = []

    # Traverse all subfolders
    for root, _, files in os.walk(folder):

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                image_path = os.path.join(root, file)

                img = cv2.imread(image_path)

                if img is None:
                    continue

                # Resize image
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Convert BGR to LAB
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

                # L channel (Input)
                L = lab[:, :, 0].astype(np.float32) / 255.0

                # AB channels (Target)
                AB = lab[:, :, 1:].astype(np.float32) / 128.0

                X.append(L.reshape(IMG_SIZE, IMG_SIZE, 1))
                Y.append(AB)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(f"Loaded {len(X)} images from {folder}")

    return X, Y