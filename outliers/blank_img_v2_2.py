import os
import gc
import cv2  # 🆕 ADDITION: OpenCV for geometric contour detection
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.ensemble import IsolationForest

# =====================================================================
# 🆕 ADDITION: GEOMETRIC CONTOUR INSPECTOR FOR CHEXPERT CUTS
# =====================================================================
def inspect_geometric_cut_opencv(img_path, black_thresh=15, min_cut_pct=0.03):
    """
    Checks if an image has artificial straight/angled black cuts or crop borders.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, "Unreadable Image"
            
        h, w = img.shape
        total_pixels = h * w

        # 1. Mask anatomy (non-black pixels)
        _, binary = cv2.threshold(img, black_thresh, 255, cv2.THRESH_BINARY)

        # 2. Find external contour of main anatomy
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return True, "Empty/Black frame"

        c = max(contours, key=cv2.contourArea)
        
        # 3. Calculate gap between Convex Hull (bounding shape) and actual anatomy contour
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(c)
        
        cut_ratio = (hull_area - contour_area) / total_pixels

        # If there's a large triangular or rectangular slice missing
        if cut_ratio >= min_cut_pct:
            return True, f"Geometric Cut Detected ({cut_ratio*100:.1f}% area slice)"
            
        return False, "Clean Geometry"
    except Exception as e:
        return False, str(e)


# =====================================================================
# 1. DUAL-NORMALIZATION PIXEL INSPECTOR
# =====================================================================
def inspect_pixel_distribution_dual(img_path, white_thresh=0.90, black_thresh=0.60):
    """
    Combines two normalizations:
    1. Absolute Bit-Depth Scaling -> Accurately flags >90% WHITE images.
    2. Min-Max Contrast Scaling   -> Accurately flags >60% BLACK/TRUNCATED images.
    """
    if not os.path.exists(img_path):
        return "CORRUPTED_OR_MISSING", "File not found"

    try:
        with Image.open(img_path) as raw_img:
            grayscale = raw_img.convert('L') if raw_img.mode in ['RGB', 'RGBA', 'P'] else raw_img
            arr = np.array(grayscale, dtype=np.float32)

        total_pixels = arr.size

        # --- NORMALIZATION A: Absolute Bit-Depth Scaling (For White Images) ---
        if arr.max() > 255.0 or raw_img.mode in ['I;16', 'I;16B', 'I;16L', 'I', 'F']:
            max_possible_value = 65535.0
        else:
            max_possible_value = 255.0
        arr_abs = (arr / max_possible_value) * 255.0

        # --- NORMALIZATION B: Min-Max Scaling (For Black/Truncated Images) ---
        img_min, img_max = float(arr.min()), float(arr.max())
        dynamic_range = img_max - img_min
        if dynamic_range > 0:
            arr_minmax = (arr - img_min) / dynamic_range * 255.0
        else:
            arr_minmax = np.zeros_like(arr)

        # Calculate pixel ratios
        white_ratio = float(np.sum(arr_abs > 235.0) / total_pixels)
        black_ratio = float(np.sum(arr_minmax < 15.0) / total_pixels)

        # 1. Mostly White Check
        if white_ratio >= white_thresh:
            return "MOSTLY_WHITE", f"{white_ratio * 100:.1f}% white (Absolute)"

        # 2. Heavy Black / Truncated Check
        if black_ratio <= black_thresh or black_ratio >= 0.05:  # Added a hard threshold for heavy black
            return "HEAVY_BLACK_TRUNCATED", f"{black_ratio * 100:.1f}% black (MinMax)"

        # 3. Candidate Cut / Anomaly
        return "STRUCTURAL_CUT_CANDIDATE", f"White: {white_ratio*100:.1f}%, Black: {black_ratio*100:.1f}%"

    except Exception as e:
        return "CORRUPTED_OR_MISSING", str(e)


# =====================================================================
# 2. MAIN OVERNIGHT MULTI-SEED PIPELINE
# =====================================================================
def main():
    metadata_path = "/work/1mxray/Xray/main_metadata_final.csv"
    features_path = "/work/1mxray/Xray/raddino_features_correct_2.npy"


    # --- EXPERIMENT CONFIGURATION ---
    NUM_SEEDS = 1           # Runs 14 distinct random seeds overnight
    SCORE_THRESHOLD = -0.05  # Lower threshold to catch subtle cut/outlier candidates
    MIN_VOTES_TO_FLAG = 1    # Keep images flagged by at least 1 seed

    # 🆕 ADDITION: CONTROL TOGGLES FOR HIGH PRECISION
    FILTER_DATASET = ""     # Set to "CheXpert" (or "ChestXpert") to run ONLY on CheXpert. Set to None for ALL datasets.
    USE_OPENCV_GEOMETRY = False      # Set True to filter out False Positives using OpenCV contour geometry.

    print(f"📖 Loading features from '{features_path}'...")
    X = np.load(features_path)  # Shape: [N, 768]
    N = X.shape[0]
    print(f" Loaded feature matrix: {X.shape}")

    df_meta = pd.read_csv(metadata_path)
    paths = df_meta['path'].tolist() if 'path' in df_meta.columns else df_meta.iloc[:, 1].tolist()

    # 🆕 ADDITION: INTRA-DATASET FILTERING (Removes Cross-Dataset Domain Shift)
    if FILTER_DATASET is not None:
        print(f"\n🎯 FILTERING FOR DATASET: '{FILTER_DATASET}' ONLY...")
        mask = df_meta['path'].str.contains(FILTER_DATASET, case=False, na=False)
        indices = np.where(mask)[0]
        X = X[indices]
        paths = [paths[i] for i in indices]
        N = X.shape[0]
        print(f"   Selected {N:,} images belonging to {FILTER_DATASET}.")

    # Pre-allocate accumulation arrays
    sum_scores = np.zeros(N, dtype=np.float64)
    flag_counts = np.zeros(N, dtype=np.int32)

    print("\n" + "="*70)
    print(f"STARTING MULTI-SEED ISOLATION FOREST ({NUM_SEEDS} Seeds)")
    print("="*70)

    for seed_idx in range(1, NUM_SEEDS + 1):
        current_seed = seed_idx * 42

        iso = IsolationForest(
            n_estimators=200,
            max_samples=256,
            max_features=0.3,
            contamination='auto',
            random_state=current_seed,
            n_jobs=-1
        )
        
        iso.fit(X)
        scores = iso.decision_function(X)

        # Accumulate metrics
        sum_scores += scores
        flagged_mask = scores < SCORE_THRESHOLD
        flag_counts[flagged_mask] += 1

        print(f"  • Seed {seed_idx:02d}/{NUM_SEEDS} (random_state={current_seed}) -> Flagged {np.sum(flagged_mask):,} anomalies")

        del iso, scores
        gc.collect()

    # Calculate average anomaly score
    avg_scores = sum_scores / NUM_SEEDS

    # Unique set of all flagged image indices
    ever_flagged_indices = np.where(flag_counts >= MIN_VOTES_TO_FLAG)[0]
    print(f"\n✅ Total UNIQUE images flagged across all {NUM_SEEDS} seeds: {len(ever_flagged_indices):,}")

    # =====================================================================
    # 3. DUAL-NORMALIZATION + GEOMETRIC CATEGORIZATION OF FLAGGED IMAGES
    # =====================================================================
    print("\n🔍 Running Pixel & Geometric Inspector on unique flagged images...")

    categorized_records = []
    mostly_white_paths = []
    heavy_black_paths = []
    cut_anomaly_paths = []
    false_positive_paths = []  # 🆕 ADDITION: Tracks valid X-rays filtered out by OpenCV

    for idx in tqdm(ever_flagged_indices, desc="Categorizing Flagged Images", unit="img"):
        img_p = paths[idx]
        cat, detail = inspect_pixel_distribution_dual(img_p, white_thresh=0.80, black_thresh=0.60)

        # 🆕 ADDITION: OPENCV CONTOUR VERIFICATION FOR STRUCTURAL CUTS
        if cat == "STRUCTURAL_CUT_CANDIDATE":
            if USE_OPENCV_GEOMETRY:
                is_cut, cut_detail = inspect_geometric_cut_opencv(img_p)
                if is_cut:
                    cat = "STRUCTURAL_CUT_ANOMALY"
                    detail = f"{detail} | {cut_detail}"
                else:
                    cat = "VALID_FEATURE_OUTLIER"  # Clean image (e.g. severe pathology/hardware)
                    detail = f"Passed Geometry Check ({detail})"
            else:
                cat = "STRUCTURAL_CUT_ANOMALY"

        record = {
            "path": img_p,
            "avg_anomaly_score": float(avg_scores[idx]),
            "times_flagged_out_of_seeds": int(flag_counts[idx]),
            "sub_category": cat,
            "details": detail
        }
        categorized_records.append(record)

        if cat == "MOSTLY_WHITE":
            mostly_white_paths.append(img_p)
        elif cat == "HEAVY_BLACK_TRUNCATED":
            heavy_black_paths.append(img_p)
        elif cat == "STRUCTURAL_CUT_ANOMALY":
            cut_anomaly_paths.append(img_p)
        elif cat == "VALID_FEATURE_OUTLIER":
            false_positive_paths.append(img_p)

    # Save detailed CSV
    df_out = pd.DataFrame(categorized_records)
    if not df_out.empty:
        df_out = df_out.sort_values(by=["times_flagged_out_of_seeds", "avg_anomaly_score"], ascending=[False, True])
        df_out.to_csv("overnight_flagged_categorized_.csv", index=False)

    # Save unique text lists
    with open("overnight_mostly_white.txt", "w") as f:
        f.write("\n".join(mostly_white_paths))

    with open("overnight_heavy_black.txt", "w") as f:
        f.write("\n".join(heavy_black_paths))

    with open("overnight_structural_cuts.txt", "w") as f:
        f.write("\n".join(cut_anomaly_paths))

    # Print Final Summary Report
    total_unique = len(ever_flagged_indices)
    print("\n" + "="*60)
    print("📊 OVERNIGHT ANOMALY PIPELINE FINAL BREAKDOWN")
    print("="*60)
    print(f"Target Dataset              : {FILTER_DATASET if FILTER_DATASET else 'ALL DATASETS'}")
    print(f"Total Unique Flagged Images : {total_unique:,}")
    print(f"⚪ Mostly White (>90%)       : {len(mostly_white_paths):,}\t({len(mostly_white_paths)/max(total_unique,1)*100:.2f}%) -> 'overnight_mostly_white.txt'")
    print(f"⬛ Heavy Black / Truncated  : {len(heavy_black_paths):,}\t({len(heavy_black_paths)/max(total_unique,1)*100:.2f}%) -> 'overnight_heavy_black.txt'")
    print(f"✂️ True Structural Cuts    : {len(cut_anomaly_paths):,}\t({len(cut_anomaly_paths)/max(total_unique,1)*100:.2f}%) -> 'overnight_structural_cuts.txt'")
    print(f"🛡️ Valid Outliers Filtered : {len(false_positive_paths):,}\t({len(false_positive_paths)/max(total_unique,1)*100:.2f}%) [False Positives Saved!]")
    print(f"📄 Detailed Master CSV      : 'overnight_flagged_categorized.csv'")
    print("="*60)


if __name__ == "__main__":
    main()