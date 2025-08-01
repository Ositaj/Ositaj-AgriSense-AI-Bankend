import os
import shutil
import random

# Set paths
base_dir = "D:/AgriSense-AI/dataset/PlantVillage"
train_dir = "D:/AgriSense-AI/dataset/train"
val_dir = "D:/AgriSense-AI/dataset/val"

# Split ratio
split_ratio = 0.8  # 80% train, 20% val

# Create output folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# For each class folder
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create class folders in train and val directories
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Move images
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, class_name, img)
        )
    for img in val_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(val_dir, class_name, img)
        )

print("✅ Dataset split completed!")
