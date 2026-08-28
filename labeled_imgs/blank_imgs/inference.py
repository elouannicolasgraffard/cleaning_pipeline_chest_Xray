import os
import joblib
import numpy as np
import pandas as pd

# =========================================================================
# CONFIGURATION
# =========================================================================
METADATA_CSV = "/work/1mxray/Xray/main_metadata_updated copy.csv"
FEATURES_PATH = "/work/1mxray/Xray/raddino_features_correct_1.npy"
MODEL_CHECKPOINT = (
    "/home/nicolasg/dev/labeled_imgs/blank_imgs/supervised_bad_detector_results/logistic_detector.pkl"
)
OUTPUT_DIR = "/home/nicolasg/dev/labeled_imgs/blank_imgs"

PROB_BAD_THRESH = 0.85  # Model must be >= 85% confident it's bad/blank
TOP_N_DISCOVERIES = 200  # Max number of top worst images to export


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  print("📂 Loading metadata and feature matrix for inference...")
  df_main = pd.read_csv(METADATA_CSV)
  main_col = next(
      (c for c in ["path", "ImageID", "filename"] if c in df_main.columns),
      df_main.columns[0],
  )

  if FEATURES_PATH.endswith(".npy"):
    features_np = np.load(FEATURES_PATH)
  else:
    features_np = torch.load(FEATURES_PATH).numpy()

  if not os.path.exists(MODEL_CHECKPOINT):
    raise FileNotFoundError(
        f"❌ Model checkpoint not found at '{MODEL_CHECKPOINT}'. Run training"
        " script first!"
    )

  print(f"📖 Loading model checkpoint from '{MODEL_CHECKPOINT}'...")
  clf = joblib.load(MODEL_CHECKPOINT)

  # --- RUN INFERENCE ON ENTIRE DATASET ---
  print("🚀 Scoring entire dataset for bad/blank characteristics...")
  all_probs = clf.predict_proba(features_np)[:, 1]
  df_main["prob_bad"] = all_probs
  df_main["flagged_bad"] = df_main["prob_bad"] >= PROB_BAD_THRESH

  # --- SAVE OUTPUTS ---
  df_sorted = df_main.sort_values(by="prob_bad", ascending=False)
  top_worst_paths = df_sorted.head(TOP_N_DISCOVERIES)[main_col]
  top_worst_file = os.path.join(
      OUTPUT_DIR, f"top_{TOP_N_DISCOVERIES}_worst_scored_paths.txt"
  )
  top_worst_paths.to_csv(top_worst_file, index=False, header=False)

  metadata_output_path = os.path.join(OUTPUT_DIR, "scored_metadata_full.csv")
  df_main.to_csv(metadata_output_path, index=False)

  print("\n" + "=" * 60)
  print("🎯 SUPERVISED BAD IMAGE INFERENCE REPORT")
  print("=" * 60)
  print(f" Total Dataset Scanned          : {len(df_main):,}")
  print(f" Total Flagged (>= 85% conf)    : {df_main['flagged_bad'].sum():,}")
  print(f" 🔝 Top {TOP_N_DISCOVERIES} Worst Paths Saved To    : {top_worst_file}")
  print(f" 💾 Full Scored CSV Saved To    : {metadata_output_path}")
  print("=" * 60)


if __name__ == "__main__":
  main()