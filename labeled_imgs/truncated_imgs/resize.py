from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_CSV = "/work/1mxray/Xray/main_metadata_final.csv"
OUTPUT_DIR = Path("/work/1mxray/Xray/corrected_img/black_borders/Image_truncated_cleaned")
SUSPICIOUS_DIR = Path("/work/1mxray/Xray/corrected_img/black_borders/Image_truncated_hard_crops_review")
OUTPUT_SUMMARY_CSV = "/work/1mxray/Xray/corrected_img/black_borders/processing_manifest.csv"

# HPC Multiprocessing & Checkpoint Settings
NUM_WORKERS = min(32, os.cpu_count() or 8)
CHUNK_SIZE = 10000  # Flushes progress to disk every 10k images
MIN_BRIGHTNESS_THRESH = 10
MIN_DIM = 300


def init_worker():
    """Prevents OpenCV thread contention across multiprocessing workers."""
    cv2.setNumThreads(1)


def deskew_and_crop_xray(path: str, min_brightness_thresh: int = MIN_BRIGHTNESS_THRESH):
    """Straightens rotated inner frames, crops outer black padding,

    and flags whether a crop is clean, untouched, or aggressively over-cropped.
    """
    if not isinstance(path, str) or not path.strip() or path.lower() == "nan":
        return None, None, "SKIPPED", "Invalid path"

    if not os.path.isfile(path):
        return None, None, "SKIPPED", "File does not exist"

    orig_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if orig_img is None:
        return None, None, "ERROR", "Unreadable/Corrupt image"

    h, w = orig_img.shape

    # 1. Segment non-black exposure
    _, thresh = cv2.threshold(orig_img, min_brightness_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return orig_img, None, "HARD_CROP", "Blank/Empty image content"

    largest_cnt = max(contours, key=cv2.contourArea)

    # 2. Compute rotation angle
    rect = cv2.minAreaRect(largest_cnt)
    (cx, cy), (rect_w, rect_h), angle = rect

    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    # 3. Rotate image if tilted
    if abs(angle) > 0.2:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            orig_img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        rotated = orig_img

    # 4. Tight-crop valid non-black content
    _, rotated_thresh = cv2.threshold(
        rotated, min_brightness_thresh, 255, cv2.THRESH_BINARY
    )
    rot_contours, _ = cv2.findContours(
        rotated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not rot_contours:
        return orig_img, rotated, "HARD_CROP", "Failed post-rotation contour isolation"

    largest_rot_cnt = max(rot_contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(largest_rot_cnt)

    cropped = rotated[ry : ry + rh, rx : rx + rw]
    ch, cw = cropped.shape

    # 5. Evaluate crop quality
    area_ratio = (cw * ch) / float(w * h)
    aspect_ratio = cw / float(ch)

    is_hard_crop = (
        (cw < MIN_DIM)
        or (ch < MIN_DIM)
        or (area_ratio < 0.60)
        or (aspect_ratio < 0.50)
        or (aspect_ratio > 1.85)
    )

    if is_hard_crop:
        status_msg = (
            f"Over-crop/Collapsed: area retained={area_ratio*100:.1f}%, "
            f"dim={cw}x{ch}, aspect={aspect_ratio:.2f}, angle={angle:.1f}°"
        )
        return orig_img, cropped, "HARD_CROP", status_msg

    is_modified = (
        (abs(angle) > 0.3)
        or (rx > 5)
        or (ry > 5)
        or (cw < w - 6)
        or (ch < h - 6)
    )

    if not is_modified:
        return None, None, "SKIPPED", "Image untouched/already clean"

    status_msg = f"Clean Crop: Rotated {angle:.2f}°, Cropped {w}x{h} -> {cw}x{ch}"
    return orig_img, cropped, "VALID_MODIFIED", status_msg


def process_single(path: str):
    """Worker task: routes files to valid or review folders based on crop severity."""
    orig_img, cropped_img, status, status_msg = deskew_and_crop_xray(path)

    if status in ["SKIPPED", "ERROR"]:
        return status, path, None, status_msg

    safe_name = path.strip("/").replace("/", "_").replace("\\", "_")

    if status == "HARD_CROP":
        orig_out = SUSPICIOUS_DIR / f"{safe_name}_ORIG.jpg"
        crop_out = SUSPICIOUS_DIR / f"{safe_name}_CROPPED.jpg"

        if orig_img is not None:
            cv2.imwrite(str(orig_out), orig_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if cropped_img is not None:
            cv2.imwrite(str(crop_out), cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return "HARD_CROP", path, str(crop_out), status_msg

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
                
                # Log everything except unreadable errors to record completion in manifest
                if status != "ERROR":
                    chunk_records.append(
                        {
                            "original_path": orig_p,
                            "processed_path": new_p if new_p else "UNTOUCHED",
                            "status": status,
                            "details": msg,
                        }
                    )

            # 2. Flush chunk directly to disk
            append_checkpoint_records(OUTPUT_SUMMARY_CSV, chunk_records)

    print("\n" + "=" * 55)
    print("PROCESSING SESSION COMPLETED")
    print("=" * 55)
    print(f"✅ Cleanly Modified & Saved        : {counts.get('VALID_MODIFIED', 0):,}")
    print(f"⚠️ Hard / Suspicious Crops Saved    : {counts.get('HARD_CROP', 0):,}")
    print(f"⏭️ Untouched / Skipped             : {counts.get('SKIPPED', 0):,}")
    print(f"❌ Unreadable / Missing            : {counts.get('ERROR', 0):,}")
    print(f"💾 Checkpoint Manifest             : {OUTPUT_SUMMARY_CSV}")
    print("=" * 55)


if __name__ == "__main__":
    main()