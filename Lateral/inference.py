import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import LinearProbe

class InferenceDataset(Dataset):
    """Dataset loader for unlabelled or inference-only features and paths."""
    def __init__(self, features, paths):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.paths = paths

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.paths[idx]

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths (adjust as needed)
    npy_features_path = "/work/1mxray/Xray/raddino_features_all_img_resized.npy"
    checkpoint = "/home/nicolasg/dev/Lateral/kfold_val_linear_not_clean/kfold_val_linear_best_model_not_clean.pth"
    output_csv = "model_predictions_output.csv"

    print("📦 Loading features for pure inference...")
    features_np = np.load(npy_features_path)
    
    # Optional: If you have a corresponding paths list or manifest
    # For demonstration, assume paths match feature indices or load from a CSV column
    dataset = InferenceDataset(features=features_np, paths=[f"img_idx_{i}" for i in range(len(features_np))])
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, pin_memory=True)

    # Load Model
    model = LinearProbe(in_features=768).to(device)
    if os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"📖 Loaded weights from '{checkpoint}'")
    else:
        raise FileNotFoundError(f"⚠️ Checkpoint '{checkpoint}' not found!")

    model.eval()
    all_preds, all_paths = [], []

    print("🚀 Running inference...")
    with torch.no_grad():
        for feats, paths in tqdm(loader, desc="Inference Progress"):
            feats = feats.to(device)
            out = model(feats)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_paths.extend(paths)

    # Map numeric predictions back to string labels
    label_map = {0.0: "Frontal", 1.0: "Lateral"}
    mapped_preds = [label_map.get(p, p) for p in all_preds]

    # Save results to CSV
    df_results = pd.DataFrame({
        "path": all_paths,
        "predicted_label": mapped_preds,
        "raw_prediction": all_preds
    })
    
    df_results.to_csv(output_csv, index=False)
    print(f"💾 Saved predictions successfully to '{output_csv}' with {len(df_results):,} records.")

if __name__ == "__main__":
    run_inference()