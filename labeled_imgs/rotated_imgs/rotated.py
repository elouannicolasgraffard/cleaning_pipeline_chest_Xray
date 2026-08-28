import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tr
from torch.utils.tensorboard import SummaryWriter
from rad_dino import RadDino

# Helper function to load image paths
def load_image_paths(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"❌ Text file not found at: {txt_path}")
    
    with open(txt_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
        
    print(f"📥 Loaded {len(paths):,} image paths from '{txt_path}'")
    return paths

# 1. Training Dataset (Applies random rotations dynamically)
class RotationDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
        
        # 0 = Normal, 1 = 90 deg, 2 = 180 deg, 3 = 270 deg
        label = torch.randint(0, 4, (1,)).item()
        
        if label == 1:
            image = image.rotate(90)
        elif label == 2:
            image = image.rotate(180)
        elif label == 3:
            image = image.rotate(270)
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 1b. Deterministic Validation Dataset (Fixed rotations for stable evaluation)
class DeterministicValidationDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        # Pre-assign deterministic rotation labels so validation stays consistent across epochs
        np.random.seed(42)
        self.labels = np.random.randint(0, 4, size=len(image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(self.labels[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
        
        if label == 1:
            image = image.rotate(90)
        elif label == 2:
            image = image.rotate(180)
        elif label == 3:
            image = image.rotate(270)
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 2. Fixed MLP Model (Isolates no_grad to backbone only)
class MLP(nn.Module):
    def __init__(self, in_features=768, hidden_dim=256):
        super().__init__()
        self.backbone = RadDino()
        
        # Freeze backbone parameters
        for param in self.backbone.model.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, x):
        # 🛡️ Keep backbone feature extraction frozen/no_grad, but let gradients flow through self.head!
        with torch.no_grad():
            self.backbone.eval()
            outputs = self.backbone.model(x)
            global_representation = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classification head outside no_grad context
        return self.head(global_representation)


def run_padchest_inference(metadata_path, model, best_model_path="best_rotation_model.pth", output_csv="padchest_rotated_flagged.csv"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found at: {metadata_path}")
        return

    print(f"📖 Reading PadChest metadata from '{metadata_path}'...")
    df = pd.read_csv(metadata_path, low_memory=False)
    if 'source' in df.columns:
        df = df[df['source'].astype(str).str.lower() == 'padchest']
        print(f"✅ Filtered metadata: Found {len(df):,} PadChest images based on 'source' column.")
    else:
        print("⚠️ Warning: 'source' column not found in metadata! Proceeding with all rows.")

    if len(df) == 0:
        print("❌ Error: No rows matched 'source == padchest'.")
        return

    possible_cols = ['path', 'file_path', 'ImageID', 'image_path', 'img_path']
    img_col = next((c for c in possible_cols if c in df.columns), df.columns[0])

    paths = df[img_col].dropna().astype(str).tolist()[:60000]
    print(f"🔍 Running inference on {len(paths):,} PadChest images...")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"✅ Loaded best model weights from '{best_model_path}'")
    else:
        print("⚠️ Warning: Best model checkpoint not found!")

    model.eval()

    transform_cfg = tr.Compose([
        tr.Resize((224, 224)),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    flagged_results = []
    angles_map = {1: 90, 2: 180, 3: 270}

    with torch.no_grad():
        for path in tqdm(paths, desc="PadChest Inference", unit="img"):
            try:
                img = Image.open(path).convert('RGB')
                tensor = transform_cfg(img).unsqueeze(0).to(device)
                
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prob, pred = probabilities.max(1)
                pred_class = pred.item()
                
                if pred_class != 0: # If flagged as rotated (90, 180, or 270)
                    flagged_results.append({
                        'path': path, 
                        'predicted_rotation_angle': angles_map[pred_class], 
                        'confidence': prob.item()
                    })
            except Exception:
                continue

    flagged_df = pd.DataFrame(flagged_results)
    flagged_df.to_csv(output_csv, index=False)
    print(f"💾 Saved flagged rotated images to '{output_csv}'")
    print(f"\n📊 Inference complete! Flagged {len(flagged_df):,} rotated images out of {len(paths):,}.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "/home/nicolasg/dev/train.txt"
    val_path = "/home/nicolasg/dev/val.txt"
    
    writer = SummaryWriter(log_dir="runs/rotation_training")

    paths = load_image_paths(data_path)
    val_paths = load_image_paths(val_path)

    transform_cfg = tr.Compose([
        tr.Resize((224, 224)),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = RotationDataset(paths, transform=transform_cfg)
    dataset_val = DeterministicValidationDataset(val_paths, transform=transform_cfg)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)

    model = MLP().to(device)
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 25
    best_model_path = "best_rotation_model.pth"
    best_val_loss = float('inf')

    # print("🚀 Starting training loop...")
    # for e in range(epochs):
    #     model.train()
    #     train_loss = 0.0
    #     correct_train = 0
    #     total_train = 0
        
    #     for images, labels in tqdm(dataloader, desc=f"Epoch {e+1}/{epochs} Training", unit="batch"):
    #         images, labels = images.to(device), labels.to(device)
            
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()  # ✨ Now works cleanly with correct gradient tracking!
    #         optimizer.step()
            
    #         train_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total_train += labels.size(0)
    #         correct_train += predicted.eq(labels).sum().item()

    #     avg_train_loss = train_loss / len(dataloader)
    #     train_acc = correct_train / total_train

    #     # Validation phase
    #     model.eval()
    #     val_loss = 0.0
    #     correct_val = 0
    #     total_val = 0
    #     with torch.no_grad():
    #         for images, labels in tqdm(dataloader_val, desc=f"Epoch {e+1} Validation", unit="batch"):
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = model(images)
    #             v_loss = criterion(outputs, labels)
    #             val_loss += v_loss.item()
                
    #             _, predicted = outputs.max(1)
    #             total_val += labels.size(0)
    #             correct_val += predicted.eq(labels).sum().item()
                
    #     avg_val_loss = val_loss / len(dataloader_val)
    #     val_acc = correct_val / total_val
        
    #     print(f"📈 Epoch {e+1} | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc*100:.1f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc*100:.1f}%)")

    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), best_model_path)
    #         print(f"🌟 New best model saved (Val Loss: {avg_val_loss:.4f})")

    # writer.close()
    # print("🏁 Training complete! Starting PadChest inference...")

    run_padchest_inference(
        metadata_path="/work/1mxray/Xray/metadata_corrected.csv", 
        model=model, 
        best_model_path=best_model_path
    )

if __name__ == "__main__":
    main()