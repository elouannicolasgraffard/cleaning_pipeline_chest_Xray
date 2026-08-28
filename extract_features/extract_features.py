import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from rad_dino import RadDino

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise SystemError("❌ CUDA is not available. Please ensure a GPU is allocated.")

device = "cuda"
torch.backends.cudnn.benchmark = True

# =====================================================================
# 1. DATASET DEFINITION (Sequential 0-to-N Row Indexing)
# =====================================================================
class XRayFeatureDataset(Dataset):
    def __init__(self, metadata_path, transform=None):
        self.transform = transform
        self.df = pd.read_csv(metadata_path, low_memory=False)
        
        # Reset index to guarantee contiguous rows [0, 1, 2, ..., N-1]
        self.df = self.df.reset_index(drop=True)
        
        # ---------------------------------------------------------
        # SMART PATH SELECTION: Processed vs. Original
        # ---------------------------------------------------------
        if 'processed_path' in self.df.columns and 'original_path' in self.df.columns:
            print("🔍 Found both 'processed_path' and 'original_path' columns. Applying conditional fallback...")
            
            # Clean string conversions for evaluation
            proc = self.df['processed_path'].fillna('').astype(str).str.strip()
            orig = self.df['original_path'].fillna('').astype(str).str.strip()
            
            # Condition for valid processed path (not empty, not 'nan', and file actually exists on disk)
            is_valid_processed = (proc != '') & (proc.str.lower() != 'nan')
            
            # Select processed_path if valid, otherwise fallback to original_path
            self.paths = np.where(is_valid_processed, self.df['processed_path'], self.df['original_path'])
        else:
            # Fallback behavior if columns are named differently
            possible_cols = ['path', 'file_path', 'image_path', 'Path']
            self.img_col = next((c for c in possible_cols if c in self.df.columns), self.df.columns[0])
            print(f"🔍 Using single path column: '{self.img_col}'")
            self.paths = self.df[self.img_col].astype(str).values

        print(f"📖 Loaded {len(self.df):,} image records from {metadata_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img_path = self.paths[i]
        
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')
        except Exception:
            image = Image.new('RGB', (518, 518))

        if self.transform:
            image = self.transform(image)

        # Returns image and its sequential row position (0 to N-1)
        return image, i


# =====================================================================
# 2. RADDINO FEATURE EXTRACTOR
# =====================================================================
class RadDinoFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = RadDino()
        for param in self.backbone.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.backbone.model(x)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # Strip [CLS] token
        global_representation = patch_tokens.mean(dim=1)    # Global Mean Patch Pooling -> [batch_size, 768]
        return global_representation


# =====================================================================
# 3. FAST & RESUMABLE EXTRACTION PIPELINE
# =====================================================================
def main():
    metadata_path = "/work/1mxray/Xray/main_metadata_final.csv"
    output_npy_path = "/work/1mxray/Xray/raddino_features_correct_2.npy"
    output_dat_path = "/work/1mxray/Xray/raddino_features_correct_2.dat"
    tracker_path = "/work/1mxray/Xray/processed_mask_qqq.npy"

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load Dataset
    dataset = XRayFeatureDataset(metadata_path=metadata_path, transform=transform)
    num_samples = len(dataset)
    feature_dim = 768
    total_shape = (num_samples, feature_dim)

    loader = DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # 2. Initialize Model
    model = RadDinoFeatureExtractor().to(device)
    model.eval()

    # 3. Create or Resume Memory-Mapped Array (Sized EXACTLY to num_samples)
    if os.path.exists(tracker_path):
        processed_mask = np.load(tracker_path)
        # Safety check in case metadata length changed
        if len(processed_mask) != num_samples:
            print("⚠️ Tracker size mismatch with CSV length! Resetting tracker.")
            processed_mask = np.zeros(num_samples, dtype=bool)
        else:
            print(f"🔄 Resuming job! {processed_mask.sum():,} / {num_samples:,} records already extracted.")
    else:
        processed_mask = np.zeros(num_samples, dtype=bool)

    features_memmap = np.memmap(output_dat_path, dtype='float32', mode='w+' if not os.path.exists(output_dat_path) else 'r+', shape=total_shape)

    print(f"\n🚀 Extracting RadDINO features for {num_samples:,} images on {torch.cuda.get_device_name(0)}...")

    # 4. Extraction Loop
    with torch.inference_mode():
        for images, batch_indices in tqdm(loader, desc="Extracting Features", unit="batch"):
            batch_indices_np = batch_indices.numpy()
            
            # Skip if all items in this batch were already extracted
            if np.all(processed_mask[batch_indices_np]):
                continue

            images = images.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                features = model(images)

            features_memmap[batch_indices_np] = features.cpu().numpy()
            processed_mask[batch_indices_np] = True

            # Periodic checkpoint flush
            if np.random.rand() < 0.02:
                features_memmap.flush()
                np.save(tracker_path, processed_mask)

    # Flush final memmap changes
    features_memmap.flush()
    np.save(tracker_path, processed_mask)

    # 5. Convert Memory-Map to final Clean .npy File
    print(f"\n💾 Finalizing and saving clean array to '{output_npy_path}'...")
    final_array = np.array(features_memmap)
    np.save(output_npy_path, final_array)

    # Cleanup temporary files
    if os.path.exists(output_dat_path):
        os.remove(output_dat_path)
    if os.path.exists(tracker_path):
        os.remove(tracker_path)

    print("\n" + "="*50)
    print("🎉 FEATURE EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total Rows in Metadata CSV: {num_samples:,}")
    print(f"Final .npy Matrix Shape   : {final_array.shape}")
    print("="*50)

if __name__ == "__main__":
    main()