import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Transfer labels from a source CSV to a target CSV based on matching paths.")
    parser.add_argument("--source", required=True, help="Path to the source CSV containing the updated labels.")
    parser.add_argument("--target", required=True, help="Path to the target CSV where the new column will be added.")
    parser.add_argument("--label_col", default="Support Devices", help="Name of the label column in the source CSV to transfer.")
    parser.add_argument("--new_col_name", default="Support Devices", help="Name of the new column to create in the target CSV.")
    parser.add_argument("--output", required=False, help="Path to save the resulting CSV. Defaults to [target_name]_with_labels.csv")
    args = parser.parse_args()

    source_csv = args.source
    target_csv = args.target
    label_col = args.label_col
    new_col_name = args.new_col_name

    if not os.path.exists(source_csv):
        print(f"❌ Error: Source CSV not found at: {source_csv}")
        return
    if not os.path.exists(target_csv):
        print(f"❌ Error: Target CSV not found at: {target_csv}")
        return

    # 1. Load source and target CSVs
    print(f"📖 Reading source CSV: '{source_csv}'...")
    df_source = pd.read_csv(source_csv, low_memory=False)

    print(f"📖 Reading target CSV: '{target_csv}'...")
    df_target = pd.read_csv(target_csv, low_memory=False)

    # 2. Auto-detect path columns
    source_path_col = next((c for c in ['Path', 'path', 'file_path', 'image_path'] if c in df_source.columns), df_source.columns[0])
    target_path_col = next((c for c in ['path', 'file_path', 'image_path', 'Path', 'Unnamed: 0'] if c in df_target.columns), df_target.columns[0])

    print(f"🔍 Source path column: '{source_path_col}'")
    print(f"🔍 Target path column: '{target_path_col}'")

    if label_col not in df_source.columns:
        raise ValueError(f"❌ Label column '{label_col}' not found in source CSV columns: {list(df_source.columns)}")

    # 3. Build lookup dictionary from source paths to labels
    print("🗺️ Building path-to-label mapping dictionary...")
    path_to_label = dict(zip(
        df_source[source_path_col].astype(str).str.strip(), 
        df_source[label_col]
    ))

    # 4. Map dictionary onto target CSV paths and handle missing entries with -1
    print("🔗 Transferring labels and setting unmapped rows to -1...")
    df_target[new_col_name] = df_target[target_path_col].astype(str).str.strip().map(path_to_label)
    df_target[new_col_name] = df_target[new_col_name].fillna(-1)

    # Compute statistics
    matched_count = (df_target[new_col_name] != -1).sum()
    missing_count = (df_target[new_col_name] == -1).sum()

    # 5. Save output
    output_path = args.output if args.output else target_csv.replace('.csv', '_with_transferred_labels.csv')
    df_target.to_csv(output_path, index=False)

    print("\n" + "="*50)
    print("LABEL TRANSFER COMPLETE")
    print("="*50)
    print(f"📊 Total rows in target CSV : {len(df_target):,}")
    print(f"✅ Successfully matched labels: {matched_count:,}")
    print(f"⚠️ Missing labels (set to -1): {missing_count:,}")
    print(f"💾 Saved updated CSV to     : '{output_path}'")
    print("="*50)

if __name__ == "__main__":
    main()