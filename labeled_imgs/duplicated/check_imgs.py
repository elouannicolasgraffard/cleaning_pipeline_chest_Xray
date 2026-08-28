from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import csv
import hashlib
import os
import pandas as pd
from PIL import Image
import pydicom
from tqdm import tqdm

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.dcm', '.tif', '.tiff', '.bmp'}

# =====================================================================
# 1. HELPER WORKERS (TOP-LEVEL FOR MULTIPROCESSING PICKLE SUPPORT)
# =====================================================================
def verify_single_image(path):
    """Checks if a file exists AND attempts to fully decode pixel data

    to detect corrupted/truncated image files. Handles DICOM and standard images.
    """
    path_str = str(path).strip()
    if not path_str or not os.path.isfile(path_str):
        return path_str, False, "Missing File"

    ext = os.path.splitext(path_str)[1].lower()
    try:
        if ext == '.dcm':
            dcm = pydicom.dcmread(path_str)
            _ = dcm.pixel_array  # Forces full pixel decompression/decoding
            return path_str, True, "OK"
        else:
            with Image.open(path_str) as img:
                img.verify()  # Header check
            with Image.open(path_str) as img:
                img.transpose(Image.FLIP_LEFT_RIGHT)  # Force full pixel decode
            return path_str, True, "OK"
    except Exception as e:
        return path_str, False, f"Corrupted ({type(e).__name__})"


def compute_hash(path):
    """Computes MD5 hash for candidate duplicate files."""
    hasher = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return path, hasher.hexdigest()
    except Exception:
        return path, None


# =====================================================================
# 2. HELPER: 2-PASS PHYSICAL CONTENT DUPLICATE DETECTOR
# =====================================================================
def check_physical_duplicates(file_paths, max_workers=32):
    print("\n📦 Check 3: Auditing physical content duplicates on disk...")
    print("   Pass 1: Grouping valid files by exact byte size...")

    size_groups = defaultdict(list)
    for path in file_paths:
        p_str = str(path).strip()
        if p_str and os.path.isfile(p_str):
            try:
                size_groups[os.path.getsize(p_str)].append(p_str)
            except Exception:
                continue

    candidate_paths = [p for paths in size_groups.values() if len(paths) > 1 for p in paths]

    if not candidate_paths:
        print("   ✅ No candidate files with identical byte sizes found.")
        return {}

    print(f"   Pass 2: Computing MD5 hashes for {len(candidate_paths):,} candidates with matching sizes...")

    hash_groups = defaultdict(list)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_hash, p): p for p in candidate_paths}
        for f in tqdm(as_completed(futures), total=len(candidate_paths), desc="   Hashing Candidates"):
            p, file_hash = f.result()
            if file_hash:
                hash_groups[file_hash].append(p)

    duplicate_groups = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}
    return duplicate_groups


# =====================================================================
# 3. MAIN AUDIT PIPELINE
# =====================================================================
def main():
    metadata_path = "/work/1mxray/Xray/metadata_when_mmimic_upload_raw.csv"
    checkpoint_csv = "audit_checkpoint.csv"

    print(f"📖 Reading '{metadata_path}'...")
    df = pd.read_csv(metadata_path, low_memory=False)

    path_col = next((c for c in ['path'] if c in df.columns), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    total_metadata_rows = len(df)

    # Normalize paths with .strip()
    df[path_col] = df[path_col].astype(str).str.strip()
    paths = df[path_col].tolist()

    print(f"Loaded {total_metadata_rows:,} records from metadata.\n")

    # -----------------------------------------------------------------
    # CHECK 1: DUPLICATE PATH STRINGS IN METADATA
    # -----------------------------------------------------------------
    print("🔍 Check 1: Verifying duplicate path strings in metadata.csv...")
    meta_duplicate_mask = df.duplicated(subset=[path_col], keep='first')
    meta_duplicate_count = meta_duplicate_mask.sum()
    unique_meta_paths = df[path_col].nunique()

    # -----------------------------------------------------------------
    # CHECK 2: FILE EXISTENCE & OPENABILITY (RESUMABLE CHECKPOINT)
    # -----------------------------------------------------------------
    print(f"\n🚀 Check 2: Verifying {total_metadata_rows:,} image files on disk...")
    valid_flags = {}
    error_reasons = {}

    # 🔄 Load existing checkpoint if present
    if os.path.exists(checkpoint_csv):
        try:
            chk_df = pd.read_csv(checkpoint_csv)
            for _, row in chk_df.iterrows():
                p = str(row['path']).strip()
                is_val = bool(row['is_valid'])
                valid_flags[p] = is_val
                if not is_val:
                    error_reasons[p] = str(row['reason'])
            print(f"🔄 Resuming! Loaded {len(valid_flags):,} previously audited records from '{checkpoint_csv}'.")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint file '{checkpoint_csv}': {e}. Starting fresh.")

    # Filter out paths that are already audited
    remaining_paths = [p for p in paths if p not in valid_flags]
    print(f"   Remaining images to audit: {len(remaining_paths):,} / {total_metadata_rows:,}")

    # Prepare checkpoint file in append mode with immediate flushing & syncing
    file_exists = os.path.exists(checkpoint_csv)
    chk_file = open(checkpoint_csv, 'a', newline='', encoding='utf-8')
    chk_writer = csv.writer(chk_file)
    if not file_exists:
        chk_writer.writerow(['path', 'is_valid', 'reason'])
        chk_file.flush()
        os.fsync(chk_file.fileno())

    MAX_WORKERS = 16  # Keep process count moderate to avoid storage deadlock

    if remaining_paths:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_path = {executor.submit(verify_single_image, p): p for p in remaining_paths}

            for future in tqdm(as_completed(future_to_path), total=len(remaining_paths), desc=" Auditing Images"):
                path = future_to_path[future]
                try:
                    p, is_valid, reason = future.result(timeout=300.0)
                    valid_flags[p] = is_valid
                    if not is_valid:
                        error_reasons[p] = reason
                    chk_writer.writerow([p, is_valid, reason])
                except TimeoutError:
                    valid_flags[path] = False
                    error_reasons[path] = "Read Timeout (>300s)"
                    chk_writer.writerow([path, False, "Read Timeout (>300s)"])
                except Exception as e:
                    valid_flags[path] = False
                    error_reasons[path] = f"Worker Error ({type(e).__name__})"
                    chk_writer.writerow([path, False, f"Worker Error ({type(e).__name__})"])

                # Immediate flush and fsync per row so progress is NEVER lost
                chk_file.flush()
                os.fsync(chk_file.fileno())

    chk_file.close()

    # Calculate validity totals
    df['is_valid'] = df[path_col].map(valid_flags).fillna(False)
    valid_count = df['is_valid'].sum()
    invalid_count = total_metadata_rows - valid_count

    # -----------------------------------------------------------------
    # CHECK 3: PHYSICAL CONTENT DUPLICATES (MD5 HASHING)
    # -----------------------------------------------------------------
    valid_paths = df[df['is_valid']][path_col].tolist()
    content_duplicates = check_physical_duplicates(valid_paths, max_workers=32)
    duplicate_extra_count = sum(len(dup_paths) - 1 for dup_paths in content_duplicates.values())

    # -----------------------------------------------------------------
    # CHECK 4: DISK DIRECTORY VS METADATA FILE COUNT
    # -----------------------------------------------------------------
    print("\n📁 Check 4: Scanning disk directory structure to compare total file count...")
    existing_paths = [p for p in paths if os.path.exists(p)]
    total_disk_images = 0
    common_root = None

    if existing_paths:
        try:
            common_root = os.path.commonpath(existing_paths)
            if os.path.isfile(common_root):
                common_root = os.path.dirname(common_root)

            print(f"   Found dataset root directory: '{common_root}'")
            print("   Counting all image files in directory...")
            for root, _, files in os.walk(common_root):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        total_disk_images += 1
        except Exception as e:
            print(f"⚠️ Could not automatically map disk directory: {e}")

    # -----------------------------------------------------------------
    # SUMMARY REPORT
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 COMPLETE DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(f"1. Metadata Entry Check:")
    print(f"   - Total Metadata Rows       : {total_metadata_rows:,}")
    print(f"   - Unique Path Entries       : {unique_meta_paths:,}")
    print(f"   - Duplicate Path Rows       : {meta_duplicate_count:,}")
    print("-" * 60)
    print(f"2. Image File Integrity:")
    print(f"   - ✅ Valid & Readable       : {valid_count:,}")
    print(f"   - ❌ Unreadable / Missing    : {invalid_count:,}")
    print("-" * 60)
    print(f"3. Content Duplicate Check:")
    print(f"   - Identical Content Groups  : {len(content_duplicates):,}")
    print(f"   - Duplicate Image Files     : {duplicate_extra_count:,}")
    print("-" * 60)
    if common_root:
        print(f"4. Directory vs Metadata Count:")
        print(f"   - Total Image Files on Disk : {total_disk_images:,}")
        print(f"   - Total Valid Metadata Files: {valid_count:,}")
        diff = total_disk_images - valid_count
        if diff == 0:
            print("   - MATCH: Every file on disk is accounted for and valid!")
        elif diff > 0:
            print(f"   - ⚠️ DISCREPANCY: {diff:,} files on disk are NOT in metadata!")
        else:
            print(f"   - ⚠️ DISCREPANCY: Metadata lists {abs(diff):,} more files than exist on disk!")
    print("=" * 60)

    # -----------------------------------------------------------------
    # CLEANING & EXPORT
    # -----------------------------------------------------------------
    duplicate_paths_to_drop = set()
    for hash_val, dup_paths in content_duplicates.items():
        for p in dup_paths[1:]:
            duplicate_paths_to_drop.add(p)

    is_content_dup = df[path_col].isin(duplicate_paths_to_drop)

    if content_duplicates:
        dup_rows = []
        for hash_val, dup_paths in content_duplicates.items():
            for p in dup_paths:
                dup_rows.append({"md5_hash": hash_val, "path": p})
        dup_df = pd.DataFrame(dup_rows)
        dup_df.to_csv("duplicate_content_images.csv", index=False)
        print(f"\n💾 Saved physical duplicate mapping to 'duplicate_content_images.csv'")

    if invalid_count > 0 or meta_duplicate_count > 0 or duplicate_extra_count > 0:
        clean_df = df[df['is_valid'] & (~meta_duplicate_mask) & (~is_content_dup)].drop(columns=['is_valid'])
        clean_df.to_csv("metadata_cleaned.csv", index=False)
        print(f"💾 Saved 100% clean metadata to 'metadata_cleaned.csv' ({len(clean_df):,} clean rows).")

        if invalid_count > 0:
            broken_df = df[~df['is_valid']].copy()
            broken_df['error_reason'] = broken_df[path_col].map(error_reasons)
            broken_df.to_csv("corrupted_or_missing_images_al_img.csv", index=False)
            print(f"💾 Saved corrupted/missing report to 'corrupted_or_missing_images_al_img.csv'.")
    else:
        print("\n🎉 Dataset is 100% clean! All images are readable, unique in content, and present.")


if __name__ == "__main__":
    main()