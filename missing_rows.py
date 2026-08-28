import pandas as pd

# =========================================================================
# CONFIGURATION: Update paths to match your files
# =========================================================================
complete_csv_path = "/work/1mxray/Xray/all_raw_images_minus_duplicated_corrupted2.csv"  # Your complete master CSV
missing_paths_txt = "/home/nicolasg/dev/bad_images.txt"  # Your text file with target paths
output_matched_csv = "extracted_missing_rows.csv"                  # Where to save the output rows

def extract_rows():
    print("📖 Loading complete dataset CSV...")
    # on_bad_lines='skip' protects against any stray comma/formatting errors in the large CSV
    df_complete = pd.read_csv(complete_csv_path, low_memory=False, on_bad_lines='skip')

    print("📖 Loading missing paths text file...")
    with open(missing_paths_txt, 'r', encoding='utf-8') as f:
        # Clean up whitespace and ignore empty lines
        missing_paths = [line.strip() for line in f if line.strip()]

    print(f"🎯 Loaded {len(missing_paths):,} target paths to search for.")

    # Dynamically find the path column name in your CSV
    path_col = next((c for c in ['path', 'ImageID', 'filename'] if c in df_complete.columns), 'path')
    print(f"🔍 Using column '{path_col}' to match paths.")

    # Use a set for ultra-fast lookup and filter the dataframe
    missing_set = set(missing_paths)
    df_matched = df_complete[df_complete[path_col].astype(str).isin(missing_set)]

    print(f"✨ Successfully matched {len(df_matched):,} corresponding rows out of {len(missing_paths):,} targets.")

    # Save the extracted rows to a new file
    df_matched.to_csv(output_matched_csv, index=False)
    print(f"💾 Saved matched rows to '{output_matched_csv}'")

if __name__ == "__main__":
    extract_rows()