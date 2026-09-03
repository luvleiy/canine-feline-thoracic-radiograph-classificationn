from pathlib import Path
import numpy as np
import cv2
import pydicom
import shutil


# ============================================================
# 1. FOLDER SETTINGS
# ============================================================

BASE_DIR = Path(r"D:\thesis\VetXRay")

# Your selected DICOM dataset
INPUT_DIR = BASE_DIR / "Selected_Dataset"

# Where the processed PNG files will be saved
OUTPUT_DIR = BASE_DIR / "Processed_Dataset"


# ============================================================
# 2. IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

# ImageNet normalization values
IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ============================================================
# 3. CHECK INPUT FOLDER
# ============================================================

if not INPUT_DIR.exists():

    print("ERROR: Selected_Dataset was not found.")
    print(f"Expected location: {INPUT_DIR}")

    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# 4. CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 5. FIND ALL DICOM FILES
# ============================================================

print("Searching for DICOM files...")

dicom_files = list(INPUT_DIR.rglob("*.dcm"))

print(f"DICOM files found: {len(dicom_files)}")


if len(dicom_files) == 0:

    print("\nERROR: No .dcm files were found.")

    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# 6. PROCESS EACH DICOM IMAGE
# ============================================================

processed = 0
failed = 0

failed_files = []


for index, dicom_path in enumerate(dicom_files, start=1):

    try:

        # ----------------------------------------------------
        # READ DICOM
        # ----------------------------------------------------

        ds = pydicom.dcmread(dicom_path)

        # Get the pixel data
        image = ds.pixel_array.astype(np.float32)


        # ----------------------------------------------------
        # HANDLE MONOCHROME1
        # ----------------------------------------------------

        # Some DICOM images use MONOCHROME1, where
        # higher values represent darker pixels.
        #
        # Invert them so the image has a consistent
        # black/white orientation.

        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":

            image = np.max(image) - image


        # ----------------------------------------------------
        # INTENSITY STANDARDIZATION
        # ----------------------------------------------------

        # Remove extreme pixel values using percentile
        # clipping.

        low = np.percentile(image, 1)
        high = np.percentile(image, 99)


        # Avoid division by zero
        if high <= low:

            low = np.min(image)
            high = np.max(image)


        if high > low:

            image = np.clip(image, low, high)

            image = (
                (image - low)
                / (high - low)
            )

        else:

            image = np.zeros_like(image)


        # ----------------------------------------------------
        # CONVERT TO 8-BIT
        # ----------------------------------------------------

        image = (image * 255).astype(np.uint8)


        # ----------------------------------------------------
        # RESIZE TO 224 × 224
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )


        # ----------------------------------------------------
        # CONVERT GRAYSCALE TO 3 CHANNELS
        # ----------------------------------------------------

        image_3channel = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )


        # ----------------------------------------------------
        # SAVE AS LOSSLESS PNG
        # ----------------------------------------------------

        # Preserve the original class folder structure.

        relative_path = dicom_path.relative_to(INPUT_DIR)

        output_path = (
            OUTPUT_DIR
            / relative_path.parent
            / (relative_path.stem + ".png")
        )


        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        cv2.imwrite(
            str(output_path),
            image_3channel,
            [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )


        processed += 1


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if processed % 100 == 0:

            print(
                f"Processed: {processed}/{len(dicom_files)}"
            )


    except Exception as error:

        failed += 1

        failed_files.append({
            "file": str(dicom_path),
            "error": str(error)
        })

        print(
            f"\nERROR processing: {dicom_path}"
        )

        print(f"Reason: {error}")


# ============================================================
# 7. SAVE FAILED FILE LIST
# ============================================================

if failed_files:

    failed_log = BASE_DIR / "preprocessing_failed_files.txt"

    with open(
        failed_log,
        "w",
        encoding="utf-8"
    ) as file:

        for item in failed_files:

            file.write(
                f"{item['file']}\n"
            )

            file.write(
                f"ERROR: {item['error']}\n\n"
            )

    print("\nFailed-file log:")
    print(failed_log)


# ============================================================
# 8. FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("             PREPROCESSING COMPLETE")
print("=" * 60)

print(f"DICOM files found:  {len(dicom_files)}")
print(f"Successfully processed: {processed}")
print(f"Failed:             {failed}")

print("\nProcessed dataset:")
print(OUTPUT_DIR)

print("=" * 60)

input("\nPress Enter to exit...")