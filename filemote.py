from pathlib import Path
import pandas as pd

# Define paths and verify they exist
base_dir = Path("/work/1mxray/Xray/datasets/VINDR")
metadata_dir = base_dir / "metadata"
images_dir = base_dir / "deid_png"
master_csv_path = Path("/work/1mxray/Xray/raw_metadata copy.csv")

print(f"Checking metadata directory: {metadata_dir}")
if not metadata_dir.exists():
  raise FileNotFoundError(f"Metadata directory does not exist: {metadata_dir}")

# Load master metadata schema
master_df = pd.read_csv(master_csv_path)
target_columns = master_df.columns.tolist()

# Safely load splits metadata with existence checks
splits = ["train", "valid", "test"]
new_dfs = []

for split in splits:
  csv_path = metadata_dir / f"{split}_metadata.csv"
  if csv_path.exists():
    print(f"Found: {csv_path.name}")
    df = pd.read_csv(csv_path)
    new_dfs.append(df)
  else:
    print(f"Warning: Could not find {csv_path}")

if not new_dfs:
  raise ValueError(
      f"No metadata CSV files were found inside {metadata_dir}. Check folder"
      " names."
  )

rex_df = pd.concat(new_dfs, ignore_index=True)


# Generate the absolute path matching the nested directory structure
def get_exact_absolute_path(row_id, study_uid):
  try:
    parts = row_id.split("_")
    hash1 = parts[0][1:]  # strip 'p'
    hash2 = parts[1][1:]  # strip 'a'

    study_path = (
        images_dir / hash1 / hash2 / "studies" / str(study_uid).strip()
    )
    if study_path.exists():
      png_files = list(study_path.rglob("*.png"))
      if png_files:
        return str(png_files[0].resolve())
  except Exception:
    pass
  return None


print("Mapping absolute paths for new images...")
rex_df["path"] = rex_df.apply(
    lambda r: get_exact_absolute_path(r["id"], r["StudyInstanceUid"]), axis=1
)
rex_df = rex_df.dropna(subset=["path"])
print(f"Successfully resolved paths for {len(rex_df)} images.")

# Create an aligned DataFrame that strictly uses the master column schema
aligned_df = pd.DataFrame(columns=target_columns)

for col in target_columns:
  if col == "path":
    aligned_df[col] = rex_df["path"]
  elif col == "sex" and "PatientSex" in rex_df.columns:
    aligned_df[col] = rex_df["PatientSex"]
  elif col == "age" and "PatientAge" in rex_df.columns:
    aligned_df[col] = rex_df["PatientAge"]
  elif col == "source":
    aligned_df[col] = "ReXGradient"
  else:
    aligned_df[col] = pd.NA

# Append cleanly to the main metadata file without altering structure
final_master_df = pd.concat([master_df, aligned_df], ignore_index=True)
final_master_df.to_csv(master_csv_path, index=False)

print(
    f"Successfully mapped and appended. New total rows:"
    f" {len(final_master_df)}"
)