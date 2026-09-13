import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A

print("=== Running Albumentations Pipeline for Model 2 ===")

# 1. Conservative transformation pipeline (strictly no horizontal flipping)
# Preserves lateralized thoracic anatomy (cardiac silhouette apex, asymmetric lung field changes)
# Updated modern Albumentations syntax
transform = A.Compose([
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=0, cval=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
])

# 2. Load the common training base (Model 1)
base_df = pd.read_csv("splits/model1_train_base_manifest.csv")
output_dir = "Augmented_Dataset/Albumentations"
os.makedirs(output_dir, exist_ok=True)

# Target sample count per class: brings minority classes up toward parity (~850)
TARGET_PER_CLASS = 850

augmented_records = []
np.random.seed(42)

for class_name, group in base_df.groupby("Class"):
    count = len(group)
    needed = TARGET_PER_CLASS - count

    if needed <= 0:
        print(f"Class '{class_name}' has {count} samples (no augmentation needed).")
        continue

    print(f"Augmenting '{class_name}': {count} real samples -> generating {needed} augmented images...")
    class_save_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_save_dir, exist_ok=True)

    # Sample with replacement from existing images to generate augmented variants
    sampled_indices = np.random.choice(group.index, size=needed, replace=True)

    for i, idx in enumerate(sampled_indices):
        row = group.loc[idx]
        img_path = row["filepath"]

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Apply transformation
        aug_data = transform(image=img)
        aug_img = aug_data["image"]

        # Quality check: verify non-empty array and dimensions
        if aug_img.shape != (224, 224, 3) or np.isnan(aug_img).any():
            continue

        # Save to disk as lossless PNG
        stem = os.path.splitext(os.path.basename(img_path))[0]
        save_filename = f"{stem}_aug_{i:04d}.png"
        save_filepath = os.path.join(class_save_dir, save_filename).replace("\\", "/")
        cv2.imwrite(save_filepath, aug_img)

        # Record augmented sample metadata
        record = row.to_dict()
        record["filepath"] = save_filepath
        record["FileName"] = save_filename
        record["is_augmented"] = True
        augmented_records.append(record)

# 3. Create Model 2 Training Manifest (Base + Validated Albumentations)
base_df["is_augmented"] = False
if augmented_records:
    aug_df = pd.DataFrame(augmented_records)
    model2_df = pd.concat([base_df, aug_df], ignore_index=True)
else:
    model2_df = base_df.copy()

model2_df.to_csv("splits/model2_train_manifest.csv", index=False)

print("\n=== Augmentation Complete ===")
print(f"Model 1 (Original Real Base): {len(base_df)} samples")
print(f"Augmented Images Generated:   {len(augmented_records)} samples")
print(f"Model 2 Total Training Base:  {len(model2_df)} samples")
print("\nModel 2 Class Distribution:")
print(model2_df["Class"].value_counts())
print("\nSaved manifest to 'splits/model2_train_manifest.csv'")