from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CSV = "/home/nicolasg/dev/labeled_imgs/white_borders/white.csv"
OUTPUT_DIR = Path("/work/1mxray/Xray/corrected_img/white_borders/Image_white_borders_cleaned")
SUSPICIOUS_DIR = Path("/work/1mxray/Xray/corrected_img/white_borders/Image_white_borders_hard_crops_review")
OUTPUT_SUMMARY_CSV = "/work/1mxray/Xray/corrected_img/white_borders/Image_white_borders_cleaned/processing_manifest.csv"

# HPC Settings
NUM_WORKERS = min(32, os.cpu_count() or 8)
CHUNK_SIZE = 10000  # Saves checkpoint to disk every 10k images
MIN_DIM = 300       # Minimum width/height threshold


def init_worker():
    """Prevents OpenCV internal thread contention across multiprocessing workers."""
    cv2.setNumThreads(1)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalizes 8-bit, 12-bit, or 16-bit radiographs to standard uint8 [0, 255]."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.dtype == np.uint8:
        return gray

    img_f = gray.astype(np.float32)
    min_v, max_v = np.min(img_f), np.max(img_f)
    if max_v - min_v < 1e-5:
        return np.zeros_like(gray, dtype=np.uint8)

    return (((img_f - min_v) / (max_v - min_v)) * 255.0).astype(np.uint8)


def detect_white_collimator_box(gray_8u: np.ndarray):
    """Calculates active thoracic bounding box by identifying white/gray collimator

    shutters on the sides and flat blank areas on the top/bottom.
    """
    h, w = gray_8u.shape

    # Downscale copy for fast and smooth profile calculation
    scale = 500.0 / max(h, w)
    sw, sh = int(w * scale), int(h * scale)
    small = cv2.resize(gray_8u, (sw, sh), interpolation=cv2.INTER_AREA)

    max_scan_x = int(sw * 0.28)  # Scan up to 28% from left/right
    max_scan_y = int(sh * 0.20)  # Scan up to 20% from top/bottom

    # --- 1. SIDES: SCAN LEFT & RIGHT BOUNDARIES ---
    # Horizontal Sobel gradient to detect sharp collimator shutter steps
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    grad_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    col_grad = np.mean(grad_x, axis=0)

    # Left boundary
    x_left_s = 0
    for x in range(max_scan_x):
        col = small[:, x]
        white_ratio = np.mean(col > 210)
        mid_ratio = np.mean((col >= 40) & (col <= 200))
        col_std = np.std(col)

        # Flag if column is white blade, flat light-gray, or devoid of mid-tones
        if white_ratio > 0.30 or mid_ratio < 0.08 or (col_std < 12.0 and np.mean(col) > 165):
            x_left_s = x + 1
        else:
            # Check for a sharp shutter transition edge
            candidates = col_grad[max(0, x - 3) : min(sw, x + 4)]
            if len(candidates) > 0 and np.max(candidates) > 12.0:
                x_left_s = x + int(np.argmax(candidates)) - 2
            break

    # Right boundary
    x_right_s = sw
    for x in range(sw - 1, sw - max_scan_x, -1):
        col = small[:, x]
        white_ratio = np.mean(col > 210)
        mid_ratio = np.mean((col >= 40) & (col <= 200))
        col_std = np.std(col)

        if white_ratio > 0.30 or mid_ratio < 0.08 or (col_std < 12.0 and np.mean(col) > 165):
            x_right_s = x - 1
        else:
            candidates = col_grad[max(0, x - 3) : min(sw, x + 4)]
            if len(candidates) > 0 and np.max(candidates) > 12.0:
                x_right_s = x - (len(candidates) - int(np.argmax(candidates))) + 2
            break

    # --- 2. TOP & BOTTOM BOUNDARIES ---
    col_start = max(0, x_left_s)
    col_end = min(sw, x_right_s)
    if col_end - col_start < int(sw * 0.3):
        col_start, col_end = 0, sw

    # Top boundary
    y_top_s = 0
    for y in range(max_scan_y):
        row = small[y, col_start:col_end]
        white_ratio = np.mean(row > 215)
        mid_ratio = np.mean((row >= 40) & (row <= 200))
        if white_ratio > 0.35 or mid_ratio < 0.06:
            y_top_s = y + 1
        else:
            break

    # Bottom boundary
    y_bottom_s = sh
    for y in range(sh - 1, sh - max_scan_y, -1):
        row = small[y, col_start:col_end]
        white_ratio = np.mean(row > 215)
        mid_ratio = np.mean((row >= 40) & (row <= 200))
        if white_ratio > 0.35 or mid_ratio < 0.06:
            y_bottom_s = y - 1
        else:
            break

    # Map coordinates back to original full resolution
    inv_scale = 1.0 / scale
    fx1 = max(0, int(x_left_s * inv_scale))
    fx2 = min(w, int(x_right_s * inv_scale))
    fy1 = max(0, int(y_top_s * inv_scale))
    fy2 = min(h, int(y_bottom_s * inv_scale))

    if fx1 >= fx2 or fy1 >= fy2:
        return 0, 0, w, h

    return fx1, fy1, fx2, fy2


def process_white_border_xray(path: str):
    """Evaluates, crops, and categorizes a single radiograph."""
    if not isinstance(path, str) or not path.strip() or path.lower() == "nan":
        return None, None, "SKIPPED", "Invalid path"

    if not os.path.isfile(path):
        return None, None, "SKIPPED", "File does not exist"

    orig_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if orig_img is None:
        return None, None, "ERROR", "Unreadable/Corrupt image"

    norm_8u = normalize_to_uint8(orig_img)
    h, w = norm_8u.shape[:2]

    x1, y1, x2, y2 = detect_white_collimator_box(norm_8u)
    cropped_img = orig_img[y1:y2, x1:x2]
    ch, cw = cropped_img.shape[:2]

    # --- Quality Evaluation ---
    area_ratio = (cw * ch) / float(w * h)
    aspect_ratio = cw / float(ch)

    is_hard_crop = (
        (cw < MIN_DIM)
        or (ch < MIN_DIM)
        or (area_ratio < 0.55)       # Lost more than 45% of original image
        or (aspect_ratio < 0.50)     # Distorted into narrow strip
        or (aspect_ratio > 1.85)
    )

    if is_hard_crop:
        status_msg = (
            f"Hard/Suspicious Crop: area={area_ratio*100:.1f}%, "
            f"dim={cw}x{ch}, aspect={aspect_ratio:.2f}"
        )
        return orig_img, cropped_img, "HARD_CROP", status_msg

    # Check if meaningfully modified (> 5 pixels trimmed on any side)
    is_modified = (x1 > 5) or (x2 < w - 5) or (y1 > 5) or (y2 < h - 5)

    if not is_modified:
        return None, None, "SKIPPED", "Untouched/Already clean"

    status_msg = f"Cleaned White Borders: {w}x{h} -> {cw}x{ch} (Crop: x=[{x1}:{x2}], y=[{y1}:{y2}])"
    return orig_img, cropped_img, "VALID_MODIFIED", status_msg


def process_single(path: str):
    """Worker task: saves to valid or review folders based on crop quality."""
    orig_img, cropped_img, status, status_msg = process_white_border_xray(path)

    if status in ["SKIPPED", "ERROR"]:
        return status, path, None, status_msg

    safe_name = path.strip("/").replace("/", "_").replace("\\", "_")

    # Hard/Suspicious crops: Save BOTH original and cropped for manual review
    if status == "HARD_CROP":
        orig_out = SUSPICIOUS_DIR / f"{safe_name}_ORIG.jpg"
        crop_out = SUSPICIOUS_DIR / f"{safe_name}_CROPPED.jpg"

        if orig_img is not None:
            cv2.imwrite(str(orig_out), normalize_to_uint8(orig_img), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if cropped_img is not None:
            cv2.imwrite(str(crop_out), normalize_to_uint8(cropped_img), [cv2.IMWRITE_JPEG_QUALITY, 95])

        return "HARD_CROP", path, str(crop_out), status_msg

    # Clean valid crops: Save the cleaned image
    out_path = OUTPUT_DIR / safe_name
    cv2.imwrite(str(out_path), cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    return "VALID_MODIFIED", path, str(out_path), status_msg


def load_checkpoint(manifest_path: str) -> set:
    """Reads existing manifest to recover previously evaluated image paths."""
    processed_paths = set()
    if os.path.isfile(manifest_path):
        try:
            prev_df = pd.read_csv(manifest_path, usecols=["original_path"], dtype=str)
            processed_paths = set(prev_df["original_path"].dropna().tolist())
            print(f"🔄 Checkpoint loaded: {len(processed_paths):,} images already processed.")
        except Exception as e:
            print(f"⚠️ Could not load prior checkpoint ({e}). Starting fresh.")
    return processed_paths


def append_checkpoint_records(manifest_path: str, records: list):
    """Appends batch results to CSV immediately to avoid data loss on crash."""
    if not records:
        return
    df_chunk = pd.DataFrame(records)
    file_exists = os.path.isfile(manifest_path)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    df_chunk.to_csv(manifest_path, mode="a", index=False, header=not file_exists)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUSPICIOUS_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(INPUT_CSV):
        print(f"❌ Input CSV not found at: {INPUT_CSV}")
        sys.exit(1)

    print(f"📖 Loading dataset from: '{INPUT_CSV}'...")
    df = pd.read_csv(INPUT_CSV, low_memory=False, dtype=str)

    path_col = next(
        (c for c in ["path", "Path", "file_path", "image_path", "Unnamed: 0"] if c in df.columns),
        df.columns[0],
    )
    print(f"🎯 Path column: '{path_col}'")

    all_paths = df[path_col].dropna().astype(str).tolist()
    total_imgs = len(all_paths)

    # 1. Recover checkpoint state
    processed_set = load_checkpoint(OUTPUT_SUMMARY_CSV)
    remaining_paths = [p for p in all_paths if p not in processed_set]

    print(f"📊 Total Dataset         : {total_imgs:,}")
    print(f"⏭️ Already Completed     : {len(processed_set):,}")
    print(f"🎯 Remaining to Process  : {len(remaining_paths):,}")

    if not remaining_paths:
        print("✅ All images are already processed!")
        return

    print(f"🚀 Running across {NUM_WORKERS} workers (Checkpoint saving every {CHUNK_SIZE:,} imgs)...")

    counts = {"VALID_MODIFIED": 0, "HARD_CROP": 0, "SKIPPED": 0, "ERROR": 0}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        for i in range(0, len(remaining_paths), CHUNK_SIZE):
            batch_paths = remaining_paths[i : i + CHUNK_SIZE]
            futures = {executor.submit(process_single, p): p for p in batch_paths}
            chunk_records = []

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Chunk {i // CHUNK_SIZE + 1}/{(len(remaining_paths) + CHUNK_SIZE - 1) // CHUNK_SIZE}",
                unit="img",
                dynamic_ncols=True,
            ):
                status, orig_p, new_p, msg = f.result()
                counts[status] = counts.get(status, 0) + 1

                if status != "ERROR":
                    chunk_records.append(
                        {
                            "original_path": orig_p,
                            "processed_path": new_p if new_p else "UNTOUCHED",
                            "status": status,
                            "details": msg,
                        }
                    )

            # Flush batch directly to disk
            append_checkpoint_records(OUTPUT_SUMMARY_CSV, chunk_records)

    print("\n" + "=" * 55)
    print("PROCESSING SESSION COMPLETED")
    print("=" * 55)
    print(f"✅ Cleanly Modified & Saved        : {counts.get('VALID_MODIFIED', 0):,}")
    print(f"⚠️ Hard / Suspicious Crops Saved    : {counts.get('HARD_CROP', 0):,}")
    print(f"⏭️ Untouched / Skipped             : {counts.get('SKIPPED', 0):,}")
    print(f"❌ Unreadable / Missing            : {counts.get('ERROR', 0):,}")
    print(f"📁 Clean Crops Directory           : {OUTPUT_DIR.resolve()}")
    print(f"📁 Hard Crop Review Directory      : {SUSPICIOUS_DIR.resolve()}")
    print(f"💾 Checkpoint Manifest             : {OUTPUT_SUMMARY_CSV}")
    print("=" * 55)


if __name__ == "__main__":
    main()