import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Remove rows from a metadata CSV based on a text file of paths.")
    parser.add_argument("--txt", required=True, help="Path to the text file containing paths to remove (one per line).")
    parser.add_argument("--csv", required=True, help="Path to the target metadata CSV file.")
    parser.add_argument("--output", required=False, help="Path to save the cleaned CSV file. If omitted, saves as '[csv_name]_cleaned.csv'.")
    args = parser.parse_args()

    txt_path = args.txt
    csv_path = args.csv
    
    if not os.path.exists(txt_path):
        print(f"❌ Error: Text file not found at: {txt_path}")
        return
    if not os.path.exists(csv_path):
        print(f"❌ Error: Metadata CSV not found at: {csv_path}")
        return

    # 1. Load paths to remove into a fast lookup set
    print(f"📖 Reading paths to remove from text file '{txt_path}'...")
    with open(txt_path, 'r') as f:
        paths_to_remove = {line.strip() for line in f if line.strip() and line.strip().lower() != 'nan'}
    print(f"📥 Loaded {len(paths_to_remove):,} unique paths targeted for removal.")

    # 2. Load target metadata CSV
    print(f"📖 Reading target metadata CSV from '{csv_path}'...")
    df = pd.read_csv(csv_path, low_memory=False)
    original_len = len(df)

    # 3. Intelligently find the path column in target metadata
    possible_cols = ['path', 'Path', 'file_path', 'image_path', 'original_path', 'processed_path']
    path_col = next((c for c in possible_cols if c in df.columns), None)
    
    if not path_col:
        # Fallback to any column name containing 'path'
        path_col = next((c for c in df.columns if 'path' in c.lower()), df.columns[0])
        
    print(f"🔍 Using metadata column '{path_col}' to match paths.")

    # 4. Filter out matching rows
    print("🧹 Filtering out targeted rows...")
    mask = df[path_col].astype(str).str.strip().isin(paths_to_remove)
    matched_count = mask.sum()
    
    df_filtered = df[~mask].copy()

    # 5. Determine output path and save
    output_path = args.output if args.output else csv_path.replace('.csv', '_cleaned.csv')
    df_filtered.to_csv(output_path, index=False)

    print("\n" + "="*50)
    print("ROW REMOVAL COMPLETE")
    print("="*50)
    print(f"📊 Original rows : {original_len:,}")
    print(f"🗑️ Removed rows  : {matched_count:,}")
    print(f"✨ Remaining rows: {len(df_filtered):,}")
    print(f"💾 Saved cleaned CSV to: '{output_path}'")
    print("="*50)

if __name__ == "__main__":
    main()