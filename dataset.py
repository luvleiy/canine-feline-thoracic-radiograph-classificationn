import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# 6 mutually exclusive target classes mapped to integer labels 0-5
CLASS_MAP = {
    'Canine No Findings': 0,
    'Canine Cardiomegaly': 1,
    'Canine Alveolar Pattern': 2,
    'Feline No Findings': 3,
    'Feline Cardiomegaly': 4,
    'Feline Alveolar Pattern': 5,
}


class ThoracicRadiographDataset(Dataset):

    def __init__(self, manifest_csv, transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.class_to_idx = CLASS_MAP

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert Windows path separators (\) to Linux path separators (/)
        image_path = row['filepath'].replace('\\', '/')

        # Read image from the verified relative path
        image = cv2.imread(image_path)

        # Give a clear error if the image cannot be found/read
        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        # OpenCV loads images as BGR, so convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transformations if provided
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Convert to float32 and scale pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert from HWC to CHW format: (3, 224, 224)
        image = torch.tensor(image).permute(2, 0, 1)

        # Standardize using ImageNet channel statistics for ResNet50
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        image = (image - mean) / std

        # Map target class string to numeric label
        label = self.class_to_idx[row['Class']]

        return image, torch.tensor(label, dtype=torch.long)