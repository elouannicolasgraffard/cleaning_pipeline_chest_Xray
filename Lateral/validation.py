import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

from model import LinearProbe, MLPProbe
from dataset import FeatureDataset, load_dataset_by_source
from utils import plot_confusion_matrix

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata_path = "/work/1mxray/Xray/main_metadata_updated.csv"
    npy_features_path = "/work/1mxray/Xray/raddino_features_all_img_resized.npy"
    checkpoint = "/home/nicolasg/dev/Lateral/kfold_val_linear_not_clean/kfold_val_linear_best_model_not_clean.pth"

    # 1. Load data
    features_np, labels, paths = load_dataset_by_source(
        metadata_path, 
        npy_features_path, 
        target_sources=['MIMIC']
    )
    full_dataset = FeatureDataset(features=features_np, labels=labels, paths=paths)
    print(f"📦 Loaded full dataset with {len(full_dataset):,} samples for inference.")

    # 2. Create DataLoader
    infloader = DataLoader(full_dataset, batch_size=1024, shuffle=False, pin_memory=True)

    print(f"📊 Running inference on 3rd Dataset (CXR8) with {len(full_dataset):,} samples...")

    # 3. Load model weights
    model = LinearProbe(in_features=768).to(device)
    if os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"📖 Loaded weights from '{checkpoint}'")
    else:
        print(f"⚠️ Checkpoint '{checkpoint}' not found!")

    model.eval()
    all_preds, all_labels, val_sample_paths = [], [], []

    with torch.no_grad():
        for feats, labels_b, filenames in tqdm(infloader, desc="Evaluating Inference"):
            feats = feats.to(device)
            labels_b = labels_b.to(device)

            out = model(feats)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_b.cpu().numpy())
            
            # Capture file paths corresponding to the batch
            val_sample_paths.extend(filenames)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    target_names = ["Frontal", "Lateral"]

    # 4. Print metrics
    print("\n" + "="*55)
    print("Inference Results on 3rd Dataset (CXR8):\n")
    print(f"Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Macro F1 : {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=2))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    print("="*55)

    plot_confusion_matrix(cm, target_names, output_filename="cxr8_inference_confusion_matrix_linear_not_clean.png")

    # =====================================================================
    # 5. EXPORT MISTAKES FOR VERIFICATION
    # =====================================================================
    mistake_indices = np.where(all_labels != all_preds)[0]
    print(f"🚨 Total mistakes found on CXR8: {len(mistake_indices):,} / {len(all_labels):,}")

    label_map = {0: "Frontal", 1: "Lateral"}
    csv_records = []
    for idx in mistake_indices:
        t_val = int(all_labels[idx])
        p_val = int(all_preds[idx])
        csv_records.append({
            "path": val_sample_paths[idx],
            "true_label": label_map.get(t_val, t_val),
            "predicted_as": label_map.get(p_val, p_val),
            "error_type": "False Positive (Predicted Lateral, Actually Frontal)" if (t_val == 0 and p_val == 1) else "False Negative (Predicted Frontal, Actually Lateral)"
        })

    if csv_records:
        df_wrong = pd.DataFrame(csv_records)
        output_csv = "cxr8_wrong_predictions_linear_not_clean.csv"
        df_wrong.to_csv(output_csv, index=False)
        print(f"💾 Saved mistake details to '{output_csv}' for verification.")
    else:
        print("🎉 No mistakes found! Perfect score.")

if __name__ == "__main__":
    run_inference()