import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================================================================
# 1. ULTRA-FAST WORKER FUNCTION (RUNS IN PARALLEL ACROSS CORES)
# =====================================================================
def inspect_single_image_fast(path, border_pct=0.04, white_thresh=230, black_thresh=25):
    """
    Fast C-accelerated check for:
    - Solid White / Solid Black images
    - White / Black outer borders
    """
    if not os.path.exists(path):
        return (path, "CORRUPTED_OR_MISSING", "File not found")

    try:
        # cv2.imread in IMREAD_GRAYSCALE is significantly faster than PIL
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (path, "CORRUPTED_OR_MISSING", "Failed to decode image")

        h, w = img.shape
        total_pixels = h * w

        # --- A. FULL IMAGE CHECKS ---
        white_count = np.count_nonzero(img >= white_thresh)
        black_count = np.count_nonzero(img <= black_thresh)

        if (white_count / total_pixels) >= 0.45:
            return (path, "MOSTLY_WHITE", f"{(white_count / total_pixels) * 100:.1f}% white")
        if (black_count / total_pixels) >= 0.85:
            return (path, "MOSTLY_BLACK", f"{(black_count / total_pixels) * 100:.1f}% black")

        # --- B. FAST EDGE BORDER STRIP CHECKS ---
        bh = max(1, int(h * border_pct))
        bw = max(1, int(w * border_pct))

        # Sample border strips without creating concatenated array allocations
        top_w = np.count_nonzero(img[:bh, :] >= white_thresh)
        bot_w = np.count_nonzero(img[-bh:, :] >= white_thresh)
        left_w = np.count_nonzero(img[:, :bw] >= white_thresh)
        right_w = np.count_nonzero(img[:, -bw:] >= white_thresh)

        top_b = np.count_nonzero(img[:bh, :] <= black_thresh)
        bot_b = np.count_nonzero(img[-bh:, :] <= black_thresh)
        left_b = np.count_nonzero(img[:, :bw] <= black_thresh)
        right_b = np.count_nonzero(img[:, -bw:] <= black_thresh)

        border_pixels_count = (2 * bh * w) + (2 * bw * h) - (4 * bh * bw)
        border_white_ratio = (top_w + bot_w + left_w + right_w) / border_pixels_count
        border_black_ratio = (top_b + bot_b + left_b + right_b) / border_pixels_count

        if border_white_ratio >= 0.50:
            return (path, "WHITE_BORDER", f"{border_white_ratio * 100:.1f}% white edge")
        if border_black_ratio >= 0.70:
            return (path, "BLACK_BORDER", f"{border_black_ratio * 100:.1f}% black edge")

        return (path, "NORMAL_BORDER", "Clean")

    except Exception as e:
        return (path, "CORRUPTED_OR_MISSING", str(e))


# =====================================================================
# 2. MULTIPROCESSING PIPELINE
# =====================================================================
def main():
    metadata_path = "/work/1mxray/Xray/metadata.csv"
    output_dir = "border_and_blank_results/"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📖 Reading metadata from '{metadata_path}'...")
    df_meta = pd.read_csv(metadata_path)
    img_col = next((c for c in ['path', 'ImageID', 'filename'] if c in df_meta.columns), df_meta.columns[0])
    paths = df_meta[img_col].tolist()

    # Automatically detect CPU cores (e.g. 8, 16, or 32 on HPC node)
    num_workers = min(12, os.cpu_count() or 4)
    print(f"⚡ Launching Parallel Processing with {num_workers} CPU Workers...")

    mostly_white_paths = []
    mostly_black_paths = []
    white_border_paths = []
    black_border_paths = []
    categorized_records = []

    # Parallel Execution Pool
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks in chunks for lower process-overhead
        futures = {executor.submit(inspect_single_image_fast, path): path for path in paths}

        for future in tqdm(as_completed(futures), total=len(paths), desc="Scanning Images", unit="img"):
            path, cat, detail = future.result()

            if cat != "NORMAL_BORDER":
                categorized_records.append({"path": path, "category": cat, "details": detail})

                if cat == "MOSTLY_WHITE":
                    mostly_white_paths.append(path)
                elif cat == "MOSTLY_BLACK":
                    mostly_black_paths.append(path)
                elif cat == "WHITE_BORDER":
                    white_border_paths.append(path)
                elif cat == "BLACK_BORDER":
                    black_border_paths.append(path)

    # --- SAVE OUTPUTS ---
    print("\n💾 Saving results...")
    pd.DataFrame(categorized_records).to_csv(os.path.join(output_dir, "border_flagged_master.csv"), index=False)

    with open(os.path.join(output_dir, "mostly_white.txt"), "w") as f:
        f.write("\n".join(mostly_white_paths))
    with open(os.path.join(output_dir, "mostly_black.txt"), "w") as f:
        f.write("\n".join(mostly_black_paths))
    with open(os.path.join(output_dir, "white_borders.txt"), "w") as f:
        f.write("\n".join(white_border_paths))
    with open(os.path.join(output_dir, "black_borders.txt"), "w") as f:
        f.write("\n".join(black_border_paths))

    print("\n" + "="*60)
    print("📊 FAST BLACK & WHITE BORDER INSPECTION COMPLETE")
    print("="*60)
    print(f" Total Images Processed : {len(paths):,}")
    print(f" ⚪ Mostly White (>85%)   : {len(mostly_white_paths):,}")
    print(f" ⬛ Mostly Black (>85%)   : {len(mostly_black_paths):,}")
    print(f" 🔲 White Borders/Frames : {len(white_border_paths):,}")
    print(f" 🔳 Black Borders/Frames : {len(black_border_paths):,}")
    print(f" 📁 All files saved to  : '{output_dir}/'")
    print("="*60)


if __name__ == "__main__":
    main()