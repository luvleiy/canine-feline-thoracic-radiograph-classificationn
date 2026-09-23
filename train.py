import os
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 6
IMAGE_SIZE = 224

BATCH_SIZE = 32

# Temporary pipeline test
# Change back to 20 after the 1-epoch test succeeds
NUM_EPOCHS = 1

LEARNING_RATE = 0.0001

# Model 1 training manifest
TRAIN_CSV = "splits/model1_train_base_manifest.csv"

# Validation manifest
VAL_CSV = "splits/val_manifest.csv"

# Output folders
MODEL_DIR = "models"
RESULTS_DIR = "results"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Canine No Findings",
    "Canine Cardiomegaly",
    "Canine Alveolar Pattern",
    "Feline No Findings",
    "Feline Cardiomegaly",
    "Feline Alveolar Pattern"
]


# Convert class names to numerical labels
CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# ============================================================
# DATASET
# ============================================================

class ThoracicRadiographDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        # Read CSV manifest
        self.data = pd.read_csv(csv_file)

        # Store transformation
        self.transform = transform

        # CSV column names
        self.path_column = "filepath"
        self.label_column = "Class"

        # Check required columns
        if self.path_column not in self.data.columns:
            raise ValueError(
                f"Missing required column: {self.path_column}"
            )

        if self.label_column not in self.data.columns:
            raise ValueError(
                f"Missing required column: {self.label_column}"
            )

        print(f"\nLoaded: {csv_file}")
        print(f"Number of images: {len(self.data)}")
        print(f"Image column: {self.path_column}")
        print(f"Class column: {self.label_column}")

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        # Get one row from the CSV
        row = self.data.iloc[index]

        # Get image path
        # Convert Windows "\" to Linux "/"
        image_path = str(
            row[self.path_column]
        ).replace("\\", "/")

        # Get class name
        class_name = str(
            row[self.label_column]
        )

        # Check that class exists
        if class_name not in CLASS_TO_INDEX:

            raise ValueError(
                f"Unknown class: '{class_name}'\n"
                f"Expected one of: {CLASS_NAMES}"
            )

        # Open image
        image = Image.open(image_path)

        # Convert grayscale to RGB
        # ResNet50 expects 3 input channels
        image = image.convert("RGB")

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        # Convert class name to numerical label
        label = CLASS_TO_INDEX[class_name]

        return image, label


# ============================================================
# IMAGE TRANSFORMATIONS
# ============================================================

# Model 1 does NOT use image augmentation.
#
# The image is only:
# 1. Resized to 224 x 224
# 2. Converted to tensor
# 3. Normalized using ImageNet values

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation uses the same basic preprocessing
val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASETS")
print("=" * 60)

train_dataset = ThoracicRadiographDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = ThoracicRadiographDataset(
    VAL_CSV,
    transform=val_transform
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nDataset summary")
print("-" * 40)

print(
    f"Training images:   {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print(f"\nUsing device: {device}")


# ============================================================
# CREATE RESNET50 MODEL
# ============================================================

print("\nCreating ResNet50 model...")

model = create_model(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_val_loss = float("inf")

history = []


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print("=" * 60)


    # ========================================================
    # TRAINING PHASE
    # ========================================================

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        # Move images and labels to device
        images = images.to(device)

        labels = labels.to(device)


        # Clear previous gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update model weights
        optimizer.step()


        # Accumulate loss
        train_loss += (
            loss.item() *
            images.size(0)
        )


        # Get predicted class
        predictions = outputs.argmax(
            dim=1
        )


        # Count correct predictions
        train_correct += (
            predictions == labels
        ).sum().item()


        # Count total samples
        train_total += labels.size(0)


    # Calculate average training loss
    train_loss /= train_total


    # Calculate training accuracy
    train_accuracy = (
        train_correct /
        train_total
    )


    # ========================================================
    # VALIDATION PHASE
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    # Disable gradient calculation
    with torch.no_grad():

        for images, labels in val_loader:

            # Move images and labels to device
            images = images.to(device)

            labels = labels.to(device)


            # Forward pass
            outputs = model(images)


            # Calculate validation loss
            loss = criterion(
                outputs,
                labels
            )


            # Accumulate validation loss
            val_loss += (
                loss.item() *
                images.size(0)
            )


            # Get predictions
            predictions = outputs.argmax(
                dim=1
            )


            # Count correct predictions
            val_correct += (
                predictions == labels
            ).sum().item()


            # Count total samples
            val_total += labels.size(0)


    # Calculate average validation loss
    val_loss /= val_total


    # Calculate validation accuracy
    val_accuracy = (
        val_correct /
        val_total
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print(
        f"Train Loss:          {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:      {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss:     {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy

    })


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        model_path = os.path.join(

            MODEL_DIR,

            "model1_best.pth"

        )

        torch.save(

            model.state_dict(),

            model_path

        )

        print(
            f"Best model saved to: {model_path}"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_path = os.path.join(

    RESULTS_DIR,

    "model1_training_history.csv"

)

history_df.to_csv(

    history_path,

    index=False

)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)

print("MODEL 1 TRAINING COMPLETE")

print("=" * 60)

print(
    f"Best model: "
    f"{MODEL_DIR}/model1_best.pth"
)

print(
    f"Training history: "
    f"{history_path}"
)