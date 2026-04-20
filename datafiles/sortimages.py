import os
import random
import shutil


# Define paths
dataset_path = "/Users/brianmason/Desktop/SafeSleep/datafiles/images"
label_path = "/Users/brianmason/Desktop/SafeSleep/datafiles/labels"

train_image_path = "/Users/brianmason/Desktop/SafeSleep/datafiles/images/train"
val_image_path = "/Users/brianmason/Desktop/SafeSleep/datafiles/images/val"
train_label_path = "/Users/brianmason/Desktop/SafeSleep/datafiles/labels/train"
val_label_path = "/Users/brianmason/Desktop/SafeSleep/datafiles/labels/val"

# Create train/val directories if they don’t exist
for path in [train_image_path, val_image_path, train_label_path, val_label_path]:
    os.makedirs(path, exist_ok=True)

# Get all image filenames
all_images = [f for f in os.listdir(dataset_path) if f.endswith(".jpg")]
random.shuffle(all_images)

# Define split ratio
split_idx = int(len(all_images) * 0.8)  # 80% for training
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Move images and corresponding labels
for image in train_images:
    shutil.move(os.path.join(dataset_path, image), os.path.join(train_image_path, image))
    shutil.move(os.path.join(label_path, image.replace(".jpg", ".txt")), os.path.join(train_label_path, image.replace(".jpg", ".txt")))

for image in val_images:
    shutil.move(os.path.join(dataset_path, image), os.path.join(val_image_path, image))
    shutil.move(os.path.join(label_path, image.replace(".jpg", ".txt")), os.path.join(val_label_path, image.replace(".jpg", ".txt")))

print("Dataset split completed!")
