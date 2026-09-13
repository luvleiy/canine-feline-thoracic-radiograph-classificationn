import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

print("=== Starting Dataset Partitioning ===")

# 1. Load Excel Metadata
EXCEL_PATH = "final_filtered_VetXRay_dataset.xlsx"
df = pd.read_excel(EXCEL_PATH)

print(f"Loaded successfully. Total records: {len(df)}")

# 2. Map verified columns from dataset
CLASS_COL = "Class"

# Create a robust grouping identifier to separate distinct animals with the same name
# e.g., Mario_Canine_Boxer vs Mario_Feline_Persian
df["Patient_Group_ID"] = (
    df["PatientName"].astype(str).str.strip() + "_" +
    df["specie"].astype(str).str.strip() + "_" +
    df["breed"].astype(str).str.strip()
)
GROUP_COL = "Patient_Group_ID"

# Clean missing entries in critical fields
df = df.dropna(subset=[CLASS_COL, GROUP_COL]).reset_index(drop=True)
print(f"Total usable records: {len(df)}")
print(f"Unique patient groups identified: {df[GROUP_COL].nunique()}")

# 3. Stratified Group Partitioning (80% Train, 10% Val, 10% Test)
# Step 3a: Extract 10% Test (1 fold out of 10)
sgkf_test = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
for train_val_idx, test_idx in sgkf_test.split(df, df[CLASS_COL], df[GROUP_COL]):
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    break

# Step 3b: Extract 10% Val from remaining 90% (1 fold out of 9 leaves ~80% Train)
sgkf_val = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=42)
for train_idx, val_idx in sgkf_val.split(train_val_df, train_val_df[CLASS_COL], train_val_df[GROUP_COL]):
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()
    break

# 4. Verify patient isolation (prevent data leakage)
assert set(train_df[GROUP_COL]).isdisjoint(set(val_df[GROUP_COL])), "Data Leakage: Overlap between Train and Val!"
assert set(train_df[GROUP_COL]).isdisjoint(set(test_df[GROUP_COL])), "Data Leakage: Overlap between Train and Test!"
assert set(val_df[GROUP_COL]).isdisjoint(set(test_df[GROUP_COL])), "Data Leakage: Overlap between Val and Test!"

print("\nGroup isolation verified: Zero patient overlap across Train, Val, and Test.")
print(f"Train samples: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%)")
print(f"Val samples:   {len(val_df)} ({len(val_df)/len(df)*100:.2f}%)")
print(f"Test samples:  {len(test_df)} ({len(test_df)/len(df)*100:.2f}%)")

print("\nClass breakdown per split:")
summary_df = pd.DataFrame({
    "Train": train_df[CLASS_COL].value_counts(),
    "Val": val_df[CLASS_COL].value_counts(),
    "Test": test_df[CLASS_COL].value_counts()
}).fillna(0).astype(int)
print(summary_df)

# 5. Export split manifests
os.makedirs("splits", exist_ok=True)
train_df.to_csv("splits/train_manifest.csv", index=False)
val_df.to_csv("splits/val_manifest.csv", index=False)
test_df.to_csv("splits/test_manifest.csv", index=False)

print("\nSplit manifests successfully saved to 'splits/' directory.")