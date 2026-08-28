import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# =========================================================================
# CONFIGURATION
# =========================================================================
METADATA_CSV = "/work/1mxray/Xray/raw_metadata.csv"
BAD_IMAGES_CSV = (
    "/home/nicolasg/dev/labeled_imgs/blank_imgs/flagged_blanks_from_features_supervised.csv"
)
FEATURES_PATH = "/work/1mxray/Xray/raddino_features_raw.npy"
MODEL_OUTPUT_PATH = (
    "/home/nicolasg/dev/labeled_imgs/blank_imgs/supervised_bad_detector_results/logistic_detector.pkl"
)


def main():
  os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

  print("📂 Loading datasets and feature matrix...")
  df_main = pd.read_csv(METADATA_CSV)
  df_bad = pd.read_csv(BAD_IMAGES_CSV)

  # Dynamically find path columns
  main_col = next(
      (c for c in ["path", "ImageID", "filename"] if c in df_main.columns),
      df_main.columns[0],
  )
  bad_col = next(
      (c for c in ["path", "ImageID", "filename"] if c in df_bad.columns),
      df_bad.columns[0],
  )

  df_main["path_str"] = df_main[main_col].astype(str)
  bad_paths_set = set(df_bad[bad_col].astype(str))

  bad_indices = df_main[df_main["path_str"].isin(bad_paths_set)].index.tolist()
  clean_indices = df_main[
      ~df_main["path_str"].isin(bad_paths_set)
  ].index.tolist()

  print(
      f"✅ Matched {len(bad_indices)} bad/blank images out of your seed list."
  )

  if len(bad_indices) == 0:
    print("❌ Error: Zero bad images matched the metadata path column.")
    return

  # Load RadDINO Features [N, 768]
  if FEATURES_PATH.endswith(".npy"):
    features_np = np.load(FEATURES_PATH)
  else:
    features_np = torch.load(FEATURES_PATH).numpy()

  # --- BUILD BALANCED TRAINING SET ---
  np.random.seed(42)
  sampled_clean_indices = np.random.choice(
      clean_indices, size=min(5000, len(clean_indices)), replace=False
  )

  train_indices = np.concatenate([bad_indices, sampled_clean_indices])
  y_train = np.zeros(len(train_indices), dtype=int)
  y_train[: len(bad_indices)] = 1

  X_train = features_np[train_indices]

  print(
      f"🧠 Training Logistic Regression classifier on {len(X_train):,} samples"
      f" ({len(bad_indices)} bad, {len(sampled_clean_indices)} clean)..."
  )

  # --- TRAIN SUPERVISED CLASSIFIER ---
  clf = LogisticRegression(
      class_weight="balanced", C=0.1, max_iter=1000, random_state=42
  )
  clf.fit(X_train, y_train)

  # --- SAVE MODEL CHECKPOINT ---
  joblib.dump(clf, MODEL_OUTPUT_PATH)
  print(f"💾 Model checkpoint successfully saved to '{MODEL_OUTPUT_PATH}'")


if __name__ == "__main__":
  main()