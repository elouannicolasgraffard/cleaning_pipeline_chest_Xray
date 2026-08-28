import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Keep one duplicate image, remove the rest from metadata CSV.")
    parser.add_argument("--duplicates_csv", required=True, help="Path to the CSV file containing md5_hash and path.")
    parser.add_argument("--csv", required=True, help="Path to the target metadata CSV file.")
    parser.add_argument("--output", required=False, help="Path to save the cleaned CSV file. If omitted, saves as '[csv_name]_cleaned.csv'.")
    args = parser.parse_args()

    dup_path = args.duplicates_csv
    csv_path = args.csv
    
    if not os.path.exists(dup_path) or not os.path.exists(csv_path):
        print("❌ Error: One or both input files do not exist.")
        return

    # 1. Load duplicates CSV
    print(f"📖 Reading duplicates file from '{dup_path}'...")
    dup_df = pd.read_csv(dup_path, low_memory=False)
    
    if 'md5_hash' not in dup_df.columns or 'path' not in dup_df.columns:
        raise ValueError("❌ Duplicates CSV must contain 'md5_hash' and 'path' columns.")

    # 2. Intelligently select which paths to REMOVE (Keep 1, delete the rest)
    print("🔍 Grouping by md5_hash to keep 1 copy and mark redundant copies for removal...")
    paths_to_remove = set()
    
    for _, group in dup_df.groupby('md5_hash'):
        paths = group['path'].dropna().astype(str).str.strip().tolist()
        if len(paths) > 1:
            # KEEP paths[0] (the first one), and add the rest to the removal list
            paths_to_remove.update(paths[1:])
            
    print(f"📥 Identified {len(paths_to_remove):,} redundant duplicate paths to remove (keeping 1 safe copy per hash group).")

    # 3. Load target metadata CSV
    print(f"📖 Reading target metadata CSV from '{csv_path}'...")
    df = pd.read_csv(csv_path, low_memory=False)
    original_len = len(df)

    # 4. Intelligently find the path column in target metadata
    possible_cols = ['path', 'file_path', 'image_path', 'Path', 'Unnamed: 0']
    path_col = next((c for c in possible_cols if c in df.columns), df.columns[0])
    print(f"🔍 Using metadata column '{path_col}' to match paths.")

    # 5. Filter out only the redundant duplicate rows
    print("🧹 Filtering out redundant rows...")
    mask = df[path_col].astype(str).str.strip().isin(paths_to_remove)
    matched_count = mask.sum()
    
    df_filtered = df[~mask].copy()

    # 6. Save the cleaned CSV
    output_path = args.output if args.output else csv_path.replace('.csv', '_cleaned.csv')
    df_filtered.to_csv(output_path, index=False)

    print("\n" + "="*50)
    print("DEDUPLICATION COMPLETE")
    print("="*50)
    print(f"📊 Original rows         : {original_len:,}")
    print(f"🗑️ Removed redundant rows : {matched_count:,} (Kept 1 per hash)")
    print(f"✨ Remaining rows        : {len(df_filtered):,}")
    print(f"💾 Saved cleaned CSV to  : '{output_path}'")
    print("="*50)

if __name__ == "__main__":
    main()