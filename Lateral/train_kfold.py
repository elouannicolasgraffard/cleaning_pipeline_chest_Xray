import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

from model import LinearProbe, MLPProbe
from dataset import load_dataset_by_source

def run_kfold():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata_path = "/work/1mxray/aa.csv"
    npy_features_path = "/work/1mxray/Xray/raddino_features_raw.npy"

    features_np, labels, paths = load_dataset_by_source(
        metadata_path, 
        npy_features_path, 
        target_sources=['CheXpert_train', 'CheXpert_valid','PadChest']
    )
    
    n_splits = 20
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\n🔁 Running {n_splits}-Fold Cross-Validation on {len(labels):,} samples...")

    # ---------------------------------------------------------
    # 1. INITIALIZE TENSORBOARD WRITER (Clean unified logs)
    # ---------------------------------------------------------
    writer = SummaryWriter(log_dir="/home/nicolasg/dev/Lateral/runs/kfold_training_linear_not_clean")

    oof_probs = np.zeros(len(labels), dtype=np.float32)
    best_fold_acc = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(features_np, labels)):
        print(f"\n--- 🚀 FOLD {fold + 1}/{n_splits} ---")
        
        X_tr, y_tr = torch.tensor(features_np[train_idx], dtype=torch.float32), torch.tensor(labels[train_idx], dtype=torch.float32)
        X_va, y_va = torch.tensor(features_np[val_idx], dtype=torch.float32), torch.tensor(labels[val_idx], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=1024, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=1024, shuffle=False)

        fold_model = LinearProbe(in_features=768).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(fold_model.parameters(), lr=0.003, weight_decay=1e-4)

        # ---------------------------------------------------------
        # 2. TRAINING LOOP (Shared "Loss/Train" tag for all folds)
        # ---------------------------------------------------------
        epochs = 30
        for ep in range(epochs):
            fold_model.train()
            running_loss = 0.0

            for feats_b, lbls_b in train_loader:
                feats_b, lbls_b = feats_b.to(device), lbls_b.to(device)
                optimizer.zero_grad()
                out = fold_model(feats_b)
                loss = criterion(out, lbls_b)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_epoch_loss = running_loss / len(train_loader)
            print(f"   Epoch [{ep+1}/{epochs}] | Train Loss: {avg_epoch_loss:.4f}")
            
            # Logs epoch loss into a single unified chart (TensorBoard overlays all folds automatically)
            writer.add_scalar("Loss/Train", avg_epoch_loss, ep)

        # ---------------------------------------------------------
        # 3. EVALUATION LOOP
        # ---------------------------------------------------------
        fold_model.eval()
        val_preds = []
        val_loss = 0.0
        
        with torch.no_grad():
            for feats_b, lbls_b in val_loader:
                feats_b, lbls_b = feats_b.to(device), lbls_b.to(device)
                out = fold_model(feats_b)
                v_loss = criterion(out, lbls_b)
                val_loss += v_loss.item()
                
                probs = torch.sigmoid(out)
                val_preds.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        oof_probs[val_idx] = np.array(val_preds)
        fold_acc = accuracy_score(labels[val_idx], (oof_probs[val_idx] > 0.5).astype(int))
        
        print(f" Fold {fold + 1} Complete | Val Loss: {avg_val_loss:.4f} | OOF Accuracy: {fold_acc*100:.2f}%")

        # ---------------------------------------------------------
        # 4. UNIFIED METRIC LOGGING (X-axis is the Fold index 0 to 4)
        # ---------------------------------------------------------
        writer.add_scalar("Loss/Validation", avg_val_loss, fold)
        writer.add_scalar("Metrics/Accuracy", fold_acc, fold)

        # ---------------------------------------------------------
        # 5. SAVE CHECKPOINTS
        # ---------------------------------------------------------
        ckpt_path = f"/home/nicolasg/dev/Lateral/kfold_val_linear_not_clean/kfold_val_linear_fold_{fold + 1}_not_clean.pth"
        torch.save(fold_model.state_dict(), ckpt_path)

        if fold_acc > best_fold_acc:
            best_fold_acc = fold_acc
            torch.save(fold_model.state_dict(), "/home/nicolasg/dev/Lateral/kfold_val_linear_not_clean/kfold_val_linear_best_model_not_clean.pth")
            print(f" 🔥 [NEW BEST FOLD] Saved best model checkpoint.")

    writer.close()
    print("\n🎉 Training complete! Clean metrics saved to TensorBoard.")

if __name__ == "__main__":
    run_kfold()