from pathlib import Path
import shutil
import pandas as pd

# ============================================================
# 1. SET YOUR MAIN DATASET FOLDER
# ============================================================

BASE_DIR = Path(r"D:\thesis\VetXRay")

# Excel file containing the final 5,303 selected records
EXCEL_FILE = BASE_DIR / "final_filtered_VetXRay_dataset.xlsx"

# Folder where the selected DICOM files will be copied
OUTPUT_DIR = BASE_DIR / "Selected_Dataset"

# Original RX folders
RX_FOLDERS = [
    BASE_DIR / "RX_1",
    BASE_DIR / "RX_2",
    BASE_DIR / "RX_3",
    BASE_DIR / "RX_4",
    BASE_DIR / "RX_5",
]


# ============================================================
# 2. CHECK THAT THE EXCEL FILE EXISTS
# ============================================================

if not EXCEL_FILE.exists():
    print("ERROR: The Excel file was not found.")
    print(f"Expected location: {EXCEL_FILE}")
    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# 3. READ THE EXCEL FILE
# ============================================================

print("Reading Excel file...")

df = pd.read_excel(EXCEL_FILE)

print(f"Excel records found: {len(df)}")


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["FileName", "Class"]

for column in required_columns:
    if column not in df.columns:
        print(f"ERROR: Column '{column}' was not found in the Excel file.")
        print("Available columns are:")
        print(df.columns.tolist())
        input("\nPress Enter to exit...")
        raise SystemExit


# ============================================================
# 5. CREATE THE OUTPUT FOLDERS
# ============================================================

print("\nCreating class folders...")

classes = [
    "Canine No Findings",
    "Canine Cardiomegaly",
    "Canine Alveolar Pattern",
    "Feline No Findings",
    "Feline Cardiomegaly",
    "Feline Alveolar Pattern",
]

for class_name in classes:
    class_folder = OUTPUT_DIR / class_name
    class_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 6. CREATE A LIST OF ALL DICOM FILES
# ============================================================

print("\nScanning RX folders for DICOM files...")
print("This may take a while depending on the number of files.")

file_index = {}

for rx_folder in RX_FOLDERS:

    if not rx_folder.exists():
        print(f"WARNING: Folder not found: {rx_folder}")
        continue

    print(f"Scanning: {rx_folder.name}")

    for file_path in rx_folder.rglob("*"):

        if file_path.is_file():

            # Store the filename in lowercase
            # so matching is not affected by uppercase/lowercase
            filename_key = file_path.name.lower()

            # Keep the first occurrence if the same filename
            # appears more than once
            if filename_key not in file_index:
                file_index[filename_key] = file_path


print(f"\nFiles found in RX folders: {len(file_index)}")


# ============================================================
# 7. COPY THE SELECTED FILES
# ============================================================

copied = 0
missing = 0
already_copied = 0

missing_records = []

print("\nStarting file selection and copying...")
print("Please do not close this window while it is running.\n")


for _, row in df.iterrows():

    filename = str(row["FileName"]).strip()
    class_name = str(row["Class"]).strip()

    # Skip empty filenames
    if not filename or filename.lower() == "nan":
        missing += 1

        missing_records.append({
            "FileName": filename,
            "Class": class_name,
            "Reason": "Empty filename"
        })

        continue

    # Find the source file
    source_file = file_index.get(filename.lower())

    # If the exact filename wasn't found,
    # also try removing the file extension
    if source_file is None:

        filename_without_extension = Path(filename).stem.lower()

        possible_matches = [
            path
            for key, path in file_index.items()
            if Path(key).stem.lower() == filename_without_extension
        ]

        if len(possible_matches) == 1:
            source_file = possible_matches[0]

    # If the file still wasn't found
    if source_file is None:

        missing += 1

        missing_records.append({
            "FileName": filename,
            "Class": class_name,
            "Reason": "File not found in RX_1 to RX_5"
        })

        continue

    # Make sure the class folder exists
    destination_folder = OUTPUT_DIR / class_name
    destination_folder.mkdir(parents=True, exist_ok=True)

    # Destination file
    destination_file = destination_folder / source_file.name

    # If it was already copied
    if destination_file.exists():
        already_copied += 1
        continue

    # Copy the file
    shutil.copy2(source_file, destination_file)

    copied += 1

    # Show progress every 100 files
    if copied % 100 == 0:
        print(f"Copied: {copied}")


# ============================================================
# 8. SAVE A LIST OF MISSING FILES
# ============================================================

if missing_records:

    missing_file = BASE_DIR / "missing_files.xlsx"

    missing_df = pd.DataFrame(missing_records)

    missing_df.to_excel(missing_file, index=False)

    print("\nA list of missing files was created:")
    print(missing_file)


# ============================================================
# 9. FINAL REPORT
# ============================================================

print("\n")
print("=" * 55)
print("              DATASET EXTRACTION COMPLETE")
print("=" * 55)

print(f"Excel records:       {len(df)}")
print(f"Files copied:        {copied}")
print(f"Missing files:       {missing}")
print(f"Already copied:      {already_copied}")

print("\nSelected dataset location:")
print(OUTPUT_DIR)

print("=" * 55)

input("\nPress Enter to exit...")