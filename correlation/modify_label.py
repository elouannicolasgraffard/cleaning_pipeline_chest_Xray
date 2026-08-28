import os
import argparse
import pandas as pd


def update_metadata_labels(txt_path, csv_path, new_label, output_csv_path=None, match_by="exact"):
    """
    Updates the 'frontal/lateral' label in metadata.csv for images listed in a .txt file.

    :param txt_path: Path to .txt file containing image paths (one path per line)
    :param csv_path: Path to input metadata.csv
    :param new_label: Label to assign ("Frontal" or "Lateral")
    :param output_csv_path: Path to save updated CSV (if None, overwrites csv_path)
    :param match_by: 'exact' (full path match) or 'basename' (matches filename only, ignoring folder directories)
    """
    # 1. Validate inputs
    new_label = new_label.capitalize()
    if new_label not in ["Frontal", "Lateral"]:
        raise ValueError("❌ `new_label` must be either 'Frontal' or 'Lateral'")

    if output_csv_path is None:
        output_csv_path = csv_path

    # 2. Read image paths from .txt file
    print(f"📖 Reading image paths from '{txt_path}'...")
    with open(txt_path, "r") as f:
        txt_paths = [line.strip() for line in f if line.strip()]

    if match_by == "basename":
        target_set = {os.path.basename(p) for p in txt_paths}
    else:
        target_set = set(txt_paths)

    print(f"   • Found {len(target_set):,} unique paths/filenames in text file.")

    # 3. Load metadata CSV
    print(f"📖 Reading metadata CSV from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    # Automatically identify path and label columns
    img_col = next((c for c in ['path', 'filename', 'filepath'] if c in df.columns), df.columns[1])
    view_col = next((c for c in ['frontal/lateral', 'view', 'ViewPosition'] if c in df.columns), None)

    if not view_col:
        raise KeyError("❌ Could not find a 'frontal/lateral' column in metadata.csv")

    print(f"   • Image Path Column: '{img_col}'")
    print(f"   • Label Column     : '{view_col}'")

    # 4. Find matching rows
    print(f"\n🔄 Matching images and setting labels to '{new_label}'...")
    if match_by == "basename":
        mask = df[img_col].astype(str).apply(os.path.basename).isin(target_set)
    else:
        mask = df[img_col].astype(str).isin(target_set)

    matched_count = mask.sum()

    if matched_count == 0:
        print("\n⚠️ WARNING: 0 matching paths were found between the .txt file and metadata.csv!")
        print("💡 Tip: If your .txt file uses full absolute paths but CSV uses relative paths,")
        print("        try running with match_by='basename'.")
        return

    # 5. Apply the new label
    df.loc[mask, view_col] = new_label

    # 6. Save updated metadata
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Successfully updated {matched_count:,} records!")
    print(f"💾 Output saved to '{output_csv_path}'")


# =====================================================================
# CONFIGURATION & RUNNER
# =====================================================================
if __name__ == "__main__":
    # --- OPTION A: Hardcode your paths here ---
    TXT_FILE = "sendfile.txt"                  # File containing paths (1 per line)
    METADATA_CSV = "/work/1mxray/Xray/metadata.csv"
    NEW_LABEL = "Lateral"                       # "Lateral" or "Frontal"
    OUTPUT_CSV = "/work/1mxray/Xray/metadata.csv"  # Set to METADATA_CSV to overwrite in-place
    MATCH_BY = "exact"                          # "exact" or "basename"

    # Run update
    update_metadata_labels(
        txt_path=TXT_FILE,
        csv_path=METADATA_CSV,
        new_label=NEW_LABEL,
        output_csv_path=OUTPUT_CSV,
        match_by=MATCH_BY
    )