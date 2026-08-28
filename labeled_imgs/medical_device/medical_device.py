import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues

def main():
    master_meta_path = "/work/1mxray/Xray/main_metadata_final.csv"
    npy_features_path = "/work/1mxray/Xray/raddino_features_correct_2.npy"
    
    output_dir = "./cleanlab_audit_results"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load master metadata and features
    print(f"📖 Reading master metadata and features...")
    df_master = pd.read_csv(master_meta_path, low_memory=False)
    features = np.load(npy_features_path)
    
    if len(df_master) != features.shape[0]:
        raise ValueError(f"❌ Master rows ({len(df_master):,}) != .npy rows ({features.shape[0]:,}).")

    # Dynamically identify path and support device label columns
    path_col = next((c for c in ['path', 'Path', 'file_path'] if c in df_master.columns), df_master.columns[0])
    support_col = next((c for c in ['support_device', 'support devices', 'Support Devices', 'support'] if c in df_master.columns), None)
    
    if not support_col:
        raise ValueError(f"❌ Could not find a support device label column in master metadata columns: {df_master.columns.tolist()}")

    print(f"🎯 Target path column: '{path_col}'")
    print(f"🎯 Target label column: '{support_col}'")

    # 2. Parse labels and filter out NaNs directly from master metadata
    def parse_label(val):
        if pd.isna(val): return np.nan
        v_str = str(val).lower().strip()
        return 1 if v_str in ['1.0', '1', 'true', 'yes'] else (0 if v_str in ['0.0', '0', 'false', 'no'] else np.nan)

    df_master['parsed_label'] = df_master[support_col].apply(parse_label)
    
    # Drop rows where the label is missing (NaN)
    df_valid = df_master.dropna(subset=['parsed_label']).copy()
    valid_indices = df_valid.index.to_numpy()
    
    y = df_valid['parsed_label'].astype(int).to_numpy()
    paths = df_valid[path_col].astype(str).to_numpy()
    
    # Slice features using the exact valid row indices (1-to-1 mapping with master metadata)
    X = features[valid_indices]

    print(f"📊 Successfully aligned {X.shape[0]:,} samples for Cleanlab analysis.")
    print(f"   - Total Positive (1): {np.sum(y == 1):,}")
    print(f"   - Total Negative (0): {np.sum(y == 0):,}")

    # 3. Complex Ensemble Model (Deep MLP + Histogram Gradient Boosting)
    print("🔄 Initializing Complex Ensemble Classifier (Deep MLP + HistGradientBoosting)...")
    
    deep_mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),  # Deeper and wider architecture
        activation='relu',
        solver='adam',
        alpha=1e-3,                          # Stronger L2 regularization for high capacity
        max_iter=80,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    hist_gb = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=63,
        random_state=42
    )

    ensemble_clf = VotingClassifier(
        estimators=[
            ('mlp', deep_mlp), 
            ('hgb', hist_gb)
        ],
        voting='soft'  # Combines predicted probabilities for Cleanlab
    )

    print("🔄 Running 5-fold cross-validated out-of-fold predictions on the ensemble...")
    pred_probs = cross_val_predict(ensemble_clf, X, y, cv=5, method='predict_proba', n_jobs=-1)

    # 4. Cleanlab issue detection
    print("🧹 Running Cleanlab `find_label_issues`...")
    issues_mask = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )

    bad_paths = paths[issues_mask]
    bad_labels = y[issues_mask]
    bad_probs = pred_probs[issues_mask, 1]

    print(f"🔍 Final Shape Verification:")
    print(f"   - Paths:  {bad_paths.shape}")
    print(f"   - Labels: {bad_labels.shape}")
    print(f"   - Probs:  {bad_probs.shape}")
    print(f"   - Issues: {issues_mask.shape}")

    # Build a clean DataFrame containing ONLY the flagged issues
    df_issues = pd.DataFrame({
        'path': bad_paths,
        'target_label': bad_labels,
        'predicted_probability': bad_probs
    })

    # 5. Separate into False Negatives and False Positives
    false_negatives = df_issues[df_issues['target_label'] == 0]
    false_positives = df_issues[df_issues['target_label'] == 1]

    print(f"   - Potential False Negatives (Labeled 0): {len(false_negatives):,}")
    print(f"   - Potential False Positives (Labeled 1): {len(false_positives):,}")

    # 6. Export separate files
    fn_csv = os.path.join(output_dir, "false_negatives_audit.csv")
    fp_csv = os.path.join(output_dir, "false_positives_audit.csv")
    fn_txt = os.path.join(output_dir, "false_negatives_paths.txt")
    fp_txt = os.path.join(output_dir, "false_positives_paths.txt")
    
    false_negatives.to_csv(fn_csv, index=False)
    false_positives.to_csv(fp_csv, index=False)
    
    with open(fn_txt, "w") as f:
        f.write("\n".join(false_negatives['path'].astype(str)))
    with open(fp_txt, "w") as f:
        f.write("\n".join(false_positives['path'].astype(str)))

    print(f"\n💾 Exported separate files to: '{output_dir}/'")
    print(f"   - 📁 {fn_csv}")
    print(f"   - 📁 {fp_csv}")
    print(f"   - 📁 {fn_txt}")
    print(f"   - 📁 {fp_txt}")

if __name__ == "__main__":
    main()