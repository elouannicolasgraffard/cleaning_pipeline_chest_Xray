import os
import pandas as pd
import numpy as np

def compare_csvs_efficient():
    old_csv_path = "/work/1mxray/Xray/raw_metadata.csv"
    new_csv_path = "/work/1mxray/Xray/main_metadata_final.csv"
    
    print("📖 Inspecting columns to save memory...")
    # Read just headers first to find exact column names without loading data
    old_cols = pd.read_csv(old_csv_path, nrows=0).columns.tolist()
    new_cols = pd.read_csv(new_csv_path, nrows=0).columns.tolist()
    
    path_col = next((c for c in ['path', 'Path', 'file_path'] if c in old_cols and c in new_cols), None)
    view_col = next((c for c in ['frontal/lateral', 'view', 'View'] if c in old_cols and c in new_cols), None)
    support_col = next((c for c in ['support_device', 'support devices', 'Support Devices', 'support'] if c in old_cols and c in new_cols), None)
    
    if not path_col:
        print("❌ Error: Could not find matching path column.")
        return

    # Select ONLY the columns we need to load into memory (saves 80%+ RAM)
    cols_to_load = [path_col]
    if view_col: cols_to_load.append(view_col)
    if support_col: cols_to_load.append(support_col)
    
    for c in ['source', 'Source', 'dataset']:
        if c in old_cols and c not in cols_to_load: cols_to_load.append(c)

    print(f"🔍 Loading only necessary columns: {cols_to_load}")
    
    df_old = pd.read_csv(old_csv_path, usecols=lambda col: col in cols_to_load, low_memory=False, on_bad_lines='skip')
    df_new = pd.read_csv(new_csv_path, usecols=lambda col: col in cols_to_load, low_memory=False, on_bad_lines='skip')
    
    print(f"📖 Creating file keys for matching...")
    df_old['file_key'] = df_old[path_col].astype(str).apply(os.path.basename)
    df_new['file_key'] = df_new[path_col].astype(str).apply(os.path.basename)
    
    print(f"📊 Merging dataframes...")
    merged = pd.merge(df_old, df_new, on='file_key', suffixes=('_old', '_new'))
    
    # Free up memory immediately
    del df_old, df_new 
    
    # =====================================================================
    # 🚫 EXCLUDE MIMIC DATASET
    # =====================================================================
    print(f"🚫 Filtering out MIMIC dataset...")
    is_mimic = merged[f"{path_col}_old"].astype(str).str.contains('mimic', case=False, na=False) | \
               merged[f"{path_col}_new"].astype(str).str.contains('mimic', case=False, na=False)
               
    for c in [col for col in merged.columns if 'source' in col.lower() or 'dataset' in col.lower()]:
        is_mimic = is_mimic | merged[c].astype(str).str.contains('mimic', case=False, na=False)
        
    merged_non_mimic = merged[~is_mimic].copy()
    
    # Free up memory
    del merged 
    print(f"✨ Analyzing {len(merged_non_mimic):,} non-MIMIC images.\n")

    print("=" * 60)
    print(" 📊 METADATA & PATH COMPARISON REPORT (EXCLUDING MIMIC)")
    print("=" * 60)

    # 1. PATH CHANGES
    old_path_col = f"{path_col}_old"
    new_path_col = f"{path_col}_new"
    path_changes = len(merged_non_mimic[merged_non_mimic[old_path_col] != merged_non_mimic[new_path_col]])
    print(f"📂 PATH MODIFICATIONS:")
    print(f"   - Images with modified paths : {path_changes:,}")

    # 2. VIEW LABEL CHANGES
    if view_col:
        old_v = f"{view_col}_old"
        new_v = f"{view_col}_new"
        
        merged_non_mimic['clean_old'] = merged_non_mimic[old_v].astype(str).str.lower().str.strip()
        merged_non_mimic['clean_new'] = merged_non_mimic[new_v].astype(str).str.lower().str.strip()
        
        def map_view(val):
            if 'lateral' in val or val == '1': return 'Lateral'
            elif 'frontal' in val or 'ap' in val or 'pa' in val or val == '0': return 'Frontal'
            return val

        merged_non_mimic['mapped_old'] = merged_non_mimic['clean_old'].apply(map_view)
        merged_non_mimic['mapped_new'] = merged_non_mimic['clean_new'].apply(map_view)

        view_changes = merged_non_mimic[merged_non_mimic['mapped_old'] != merged_non_mimic['mapped_new']]
        lat_to_front = len(merged_non_mimic[(merged_non_mimic['mapped_old'] == 'Lateral') & (merged_non_mimic['mapped_new'] == 'Frontal')])
        front_to_lat = len(merged_non_mimic[(merged_non_mimic['mapped_old'] == 'Frontal') & (merged_non_mimic['mapped_new'] == 'Lateral')])
        other_view_changes = len(view_changes) - (lat_to_front + front_to_lat)

        print(f"\n🖼️ VIEW LABEL CHANGES (Non-MIMIC):")
        print(f"   - Total view changes         : {len(view_changes):,}")
        print(f"   - Lateral ➔ Frontal changes  : {lat_to_front:,}")
        print(f"   - Frontal ➔ Lateral changes  : {front_to_lat:,}")
        if other_view_changes > 0:
            print(f"   - Other view transitions     : {other_view_changes:,}")

    # 3. SUPPORT DEVICE CHANGES
    if support_col:
        old_s = f"{support_col}_old"
        new_s = f"{support_col}_new"
        
        support_changes = merged_non_mimic[merged_non_mimic[old_s].fillna('') != merged_non_mimic[new_s].fillna('')]
        print(f"\n🩺 SUPPORT DEVICE CHANGES (Non-MIMIC):")
        print(f"   - Total support device changes: {len(support_changes):,}")
        
        try:
            old_b = merged_non_mimic[old_s].astype(float)
            new_b = merged_non_mimic[new_s].astype(float)
            neg_to_pos = len(merged_non_mimic[(old_b == 0.0) & (new_b == 1.0)])
            pos_to_neg = len(merged_non_mimic[(old_b == 1.0) & (new_b == 0.0)])
            print(f"   - None/0 ➔ Device/1 changes   : {neg_to_pos:,}")
            print(f"   - Device/1 ➔ None/0 changes   : {pos_to_neg:,}")
        except:
            pass

    print("=" * 60)

if __name__ == "__main__":
    compare_csvs_efficient()