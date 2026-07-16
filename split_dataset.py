import os
import random
import shutil

# Path to your downloaded images
SOURCE_FOLDER = "dataset/all_images"

# Output folders
TRAIN_FOLDER = "dataset/train"
VAL_FOLDER = "dataset/validation"
TEST_FOLDER = "dataset/test"

# Create folders if they don't exist
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(VAL_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# Supported image formats
extensions = (".jpg", ".jpeg", ".png")

# Collect image files
images = [
    f for f in os.listdir(SOURCE_FOLDER)
    if f.lower().endswith(extensions)
]

# Shuffle images randomly
random.shuffle(images)

# Split ratios
train_ratio = 0.8
val_ratio = 0.1

total = len(images)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

def copy_images(image_list, destination):
    for image in image_list:
        shutil.copy(
            os.path.join(SOURCE_FOLDER, image),
            os.path.join(destination, image)
        )

copy_images(train_images, TRAIN_FOLDER)
copy_images(val_images, VAL_FOLDER)
copy_images(test_images, TEST_FOLDER)

print(f"Total Images      : {total}")
print(f"Training Images   : {len(train_images)}")
print(f"Validation Images : {len(val_images)}")
print(f"Testing Images    : {len(test_images)}")
print("Dataset split completed!")