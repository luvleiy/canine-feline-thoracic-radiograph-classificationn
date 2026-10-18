import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def crop_black_borders(img, threshold=10):
    """Crops out pure black padding and examination table collimation edges."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if not np.any(mask):
        return img  # fallback if entirely dark
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Ensure minimum crop area to avoid degenerate crops
    if (rmax - rmin > 20) and (cmax - cmin > 20):
        return img[rmin:rmax+1, cmin:cmax+1]
    return img

def apply_clahe(img):
    """Standardizes dynamic range and contrast across disparate X-ray machines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def process_dataset(manifest_csv="splits/model1_train_base_manifest.csv",
                    output_dir="data_preprocessed_cgan",
                    target_size=(224, 224)):
    df = pd.read_csv(manifest_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    new_rows = []
    print(f"Preprocessing {len(df)} base training images...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        raw_path = row['filepath']
        norm_path = raw_path.replace('\\', '/')
        if not os.path.exists(norm_path):
            continue
            
        img = cv2.imread(norm_path)
        if img is None:
            continue
            
        # 1. Remove peripheral black zero-spikes
        cropped = crop_black_borders(img, threshold=12)
        
        # 2. Standardize local contrast with CLAHE
        enhanced = apply_clahe(cropped)
        
        # 3. Resize to 224x224
        final_img = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
        
        # Save preprocessed image preserving class folder structure
        class_folder = os.path.join(output_dir, str(row['Class']))
        os.makedirs(class_folder, exist_ok=True)
        filename = os.path.basename(norm_path)
        save_path = os.path.join(class_folder, filename)
        cv2.imwrite(save_path, final_img)
        
        new_rows.append({
            'filepath': save_path,
            'Class': row['Class']
        })
        
    out_manifest = "splits/cgan_preprocessed_train_manifest.csv"
    pd.DataFrame(new_rows).to_csv(out_manifest, index=False)
    print(f"Preprocessing complete. Manifest saved to {out_manifest} with {len(new_rows)} images.")

if __name__ == '__main__':
    process_dataset()