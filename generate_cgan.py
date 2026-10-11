import os
import cv2
import numpy as np
import pandas as pd
import torch
from models_cgan import Generator

# Map text classes to the integer indices used by the Generator
CLASS_MAP = {
    'Canine No Findings': 0,
    'Canine Cardiomegaly': 1,
    'Canine Alveolar Pattern': 2,
    'Feline No Findings': 3,
    'Feline Cardiomegaly': 4,
    'Feline Alveolar Pattern': 5,
}

def generate_synthetic_dataset(checkpoint_path="checkpoints/cgan_generator_epoch_1.pth", target_per_class=850, latent_dim=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Generator onto device: {device}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Make sure train_cgan.py has completed.")

    # Initialize Generator and load weights
    netG = Generator(num_classes=6, latent_dim=latent_dim).to(device)
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
    netG.eval()

    base_df = pd.read_csv("splits/model1_train_base_manifest.csv")
    output_dir = "Augmented_Dataset/cGAN"
    os.makedirs(output_dir, exist_ok=True)

    generated_records = []
    
    with torch.no_grad():
        for class_name, group in base_df.groupby("Class"):
            current_count = len(group)
            needed = target_per_class - current_count
            
            if needed <= 0:
                print(f"Class '{class_name}' already has {current_count} samples (no cGAN generation needed).")
                continue

            print(f"Synthesizing '{class_name}': generating {needed} synthetic images...")
            class_save_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_save_dir, exist_ok=True)
            
            class_idx = CLASS_MAP[class_name]

            # Generate in mini-batches to conserve memory
            batch_size = 32
            num_batches = (needed + batch_size - 1) // batch_size
            generated_so_far = 0

            for b in range(num_batches):
                current_batch_size = min(batch_size, needed - generated_so_far)
                z = torch.randn(current_batch_size, latent_dim, device=device)
                labels = torch.full((current_batch_size,), class_idx, dtype=torch.long, device=device)
                
                # Generator output is [-1, 1]
                fake_imgs = netG(z, labels).cpu().numpy()
                
                # Denormalize from [-1, 1] to [0, 255] RGB uint8
                fake_imgs = ((fake_imgs + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                fake_imgs = np.transpose(fake_imgs, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)

                for img_arr in fake_imgs:
                    filename = f"cgan_{class_name.lower().replace(' ', '_')}_{generated_so_far:04d}.png"
                    filepath = os.path.join(class_save_dir, filename).replace("\\", "/")
                    
                    # Convert RGB to BGR for cv2.imwrite
                    bgr_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, bgr_img)

                    # Build metadata record
                    generated_records.append({
                        'filepath': filepath,
                        'FileName': filename,
                        'Class': class_name,
                        'Species': 'Canine' if 'Canine' in class_name else 'Feline',
                        'is_augmented': True,
                        'augmentation_type': 'cGAN'
                    })
                    generated_so_far += 1

    # Assemble Model 3 Training Manifest (Real Base + cGAN Synthetic)
    base_df['is_augmented'] = False
    base_df['augmentation_type'] = 'None'
    
    if generated_records:
        gen_df = pd.DataFrame(generated_records)
        model3_df = pd.concat([base_df, gen_df], ignore_index=True)
    else:
        model3_df = base_df.copy()

    model3_df.to_csv("splits/model3_train_manifest.csv", index=False)

    print("\n=== cGAN Synthesis Complete ===")
    print(f"Model 1 (Real Base):      {len(base_df)} samples")
    print(f"cGAN Images Generated:    {len(generated_records)} samples")
    print(f"Model 3 Total Training:   {len(model3_df)} samples")
    print("\nModel 3 Class Distribution:")
    print(model3_df["Class"].value_counts())
    print("\nSaved manifest to 'splits/model3_train_manifest.csv'")

if __name__ == '__main__':
    generate_synthetic_dataset()