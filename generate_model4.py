import os
import pandas as pd

SPLITS_DIR = "splits"

# 1. Paths to existing manifests
m1_path = os.path.join(SPLITS_DIR, "model1_train_base_manifest.csv")
m2_path = os.path.join(SPLITS_DIR, "model2_train_manifest.csv")
m3_path = os.path.join(SPLITS_DIR, "model3_train_manifest.csv")

# 2. Load the datasets
df_base = pd.read_csv(m1_path)
df_m2 = pd.read_csv(m2_path)
df_m3 = pd.read_csv(m3_path)

# Ensure base metadata consistency
df_base["is_augmented"] = False
df_base["augmentation_type"] = "None"

# 3. Extract only augmented/synthetic samples to avoid duplicating base images
df_albumentations = df_m2[df_m2["is_augmented"] == True].copy()
df_cgan = df_m3[df_m3["is_augmented"] == True].copy()

# Ensure explicit tagging
df_albumentations["augmentation_type"] = "Albumentations"
df_cgan["augmentation_type"] = "cGAN"

print(f"Base Real Images:          {len(df_base)}")
print(f"Albumentations Images:     {len(df_albumentations)}")
print(f"cGAN Synthetic Images:     {len(df_cgan)}")

# 4. Merge into Model 4 (Hybrid: Base + Classical + Generative)
df_model4 = pd.concat([df_base, df_albumentations, df_cgan], ignore_index=True)

# 5. Standardize file paths to forward slashes for cross-platform compatibility
df_model4["filepath"] = df_model4["filepath"].str.replace("\\", "/", regex=False)

# 6. Save the manifest
output_path = os.path.join(SPLITS_DIR, "model4_train_manifest.csv")
df_model4.to_csv(output_path, index=False)

print(f"\n--- Model 4 Manifest Created ---")
print(f"Total Rows:                {len(df_model4)}")
print(f"Output File:               {output_path}")

# 7. Print class breakdown
print("\nClass Distribution in Model 4:")
print(df_model4["Class"].value_counts())

print("\nAugmentation Breakdown in Model 4:")
print(df_model4["augmentation_type"].value_counts())