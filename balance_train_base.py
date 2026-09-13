import os
import numpy as np
import pandas as pd

print("=== Creating Common Original Training Base (Undersampling) ===")

train_df = pd.read_csv("splits/train_manifest.csv")
print("\nOriginal Training Set Distribution:")
print(train_df["Class"].value_counts())

# Cap the majority 'No Findings' category to prevent class starvation
TARGET_NO_FINDINGS = 1000

balanced_dfs = []
np.random.seed(42)

for class_name, group in train_df.groupby("Class"):
  if "No Findings" in class_name and len(group) > TARGET_NO_FINDINGS:
    sampled = group.sample(n=TARGET_NO_FINDINGS, random_state=42)
    balanced_dfs.append(sampled)
    print(f"Undersampled '{class_name}' from {len(group)} to {len(sampled)}")
  else:
    balanced_dfs.append(group)
    print(f"Retained '{class_name}': {len(group)} samples")

model1_base_df = pd.concat(balanced_dfs, ignore_index=True)

print("\nModel 1 Base Training Distribution:")
print(model1_base_df["Class"].value_counts())
print(f"Total Base Training Samples: {len(model1_base_df)}")

# Save manifest
os.makedirs("splits", exist_ok=True)
model1_base_df.to_csv("splits/model1_train_base_manifest.csv", index=False)
print("\nSaved manifest to 'splits/model1_train_base_manifest.csv'")