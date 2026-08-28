import pandas as pd
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
import os

# ============================================================
# 1. LOAD RAW CSVs — adjust paths as needed
# ============================================================
train_chexpert = pd.read_csv("/work/1mxray/Chestxpert_dataset/CheXpert_Dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/train.csv")
val_chexpert = pd.read_csv("/work/1mxray/Chestxpert_dataset/CheXpert_Dataset/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/valid.csv")
padchest = pd.read_csv("/work/1mxray/Padchest_dataset/padchest_labels.csv")
cxr8 = pd.read_csv("/work/1mxray/CXR8_dataset/info/Data_Entry_2017.csv")

# ============================================================
# 2. CHEXPERT
# ============================================================
chexpert_out = pd.DataFrame()
chexpert_out["path"] = "/work/1mxray/Xray/datasets/" +train_chexpert["Path"]          # adjust prefix if needed (e.g. join with dataset root)
chexpert_out["sex"] = train_chexpert["Sex"]
chexpert_out["age"] = train_chexpert["Age"]
chexpert_out["frontal/lateral"] = train_chexpert["Frontal/Lateral"]
chexpert_out["ap/pa"] = train_chexpert["AP/PA"]
chexpert_out["original_width"] = np.nan          # not published for CheXpert
chexpert_out["original_height"] = np.nan
chexpert_out["original_spacing"] = np.nan        # lost in DICOM->JPG conversion
chexpert_out["source"] = "CheXpert_train"


chexpert_out2 = pd.DataFrame()
chexpert_out2["path"] ="/work/1mxray/Xray/datasets/" + val_chexpert["Path"]          # adjust prefix if needed (e.g. join with dataset root)
chexpert_out2["sex"] = val_chexpert["Sex"]
chexpert_out2["age"] = val_chexpert["Age"]
chexpert_out2["frontal/lateral"] = val_chexpert["Frontal/Lateral"]
chexpert_out2["ap/pa"] = val_chexpert["AP/PA"]
chexpert_out2["original_width"] = np.nan          # not published for CheXpert
chexpert_out2["original_height"] = np.nan
chexpert_out2["original_spacing"] = np.nan        # lost in DICOM->JPG conversion
chexpert_out2["source"] = "CheXpert_valid"
# ============================================================
# 3. PADCHEST
# ============================================================
padchest_out = pd.DataFrame()
# Build full path — adjust root dir to match your HPC layout
PADCHEST_ROOT = "/work/1mxray/Xray/datasets/Padchest"
padchest_out["path"] ="/work/1mxray/Xray/datasets/Padchest/" + padchest["ImageID"].astype(str)
padchest_out["sex"] = padchest["PatientSex_DICOM"]

# Age = StudyDate - PatientBirth
study_date = pd.to_datetime(padchest["StudyDate_DICOM"], format="%Y%m%d", errors="coerce")
birth_year = pd.to_datetime(padchest["PatientBirth"], format="%Y", errors="coerce")
padchest_out["age"] = (study_date - birth_year).dt.days // 365

# Frontal/Lateral derived from Projection
def padchest_frontal_lateral(proj):
    if pd.isna(proj):
        return np.nan
    proj = str(proj).upper()
    if "L" in proj and "PA" not in proj and "AP" not in proj:
        return "Lateral"
    elif "AP" in proj or "PA" in proj:
        return "Frontal"
    return np.nan

padchest_out["frontal/lateral"] = padchest["Projection"].apply(padchest_frontal_lateral)
padchest_out["ap/pa"] = padchest["Projection"]

padchest_out["original_width"] = padchest["Columns_DICOM"]
padchest_out["original_height"] = padchest["Rows_DICOM"]
padchest_out["original_spacing"] = padchest["SpatialResolution_DICOM"]
padchest_out["source"] = "PadChest"

# ============================================================
# 4. CXR8
# ============================================================
cxr8_out = pd.DataFrame()
cxr8_out["path"] ="/work/1mxray/Xray/datasets/CXR8/" + cxr8["Image Index"]
cxr8_out["sex"] = cxr8["Patient Gender"]
cxr8_out["age"] = cxr8["Patient Age"]

def cxr8_frontal_lateral(view):
    if pd.isna(view):
        return np.nan
    return "Lateral" if str(view).upper() == "L" else "Frontal"

cxr8_out["frontal/lateral"] = cxr8["View Position"].apply(cxr8_frontal_lateral)
cxr8_out["ap/pa"] = cxr8["View Position"]

# OriginalImage[Width,Height] is often stored as two separate columns or one string — check your actual CSV
# If it's two columns already split, use them directly:
if "OriginalImageWidth" in cxr8.columns:
    cxr8_out["original_width"] = cxr8["OriginalImageWidth"]
    cxr8_out["original_height"] = cxr8["OriginalImageHeight"]
else:
    # fallback if it's a combined string column
    cxr8_out["original_width"] = np.nan
    cxr8_out["original_height"] = np.nan

if "OriginalImagePixelSpacing_x" in cxr8.columns:
    cxr8_out["original_spacing"] = cxr8["OriginalImagePixelSpacing_x"].astype(str) + "," + cxr8["OriginalImagePixelSpacing_y"].astype(str)
else:
    cxr8_out["original_spacing"] = np.nan

cxr8_out["source"] = "CXR8"
# ============================================================
# 5. COMBINE
# ============================================================
combined = pd.concat([chexpert_out,chexpert_out2, padchest_out, cxr8_out], ignore_index=True)

# ============================================================
# 6. COMPUTE "image size (now)" — SKIPPED, not computed
# ============================================================
# IMAGE_ROOT = "/path/to/dataset/root"

# def get_image_size(path):
#     full_path = path if os.path.isabs(path) else os.path.join(IMAGE_ROOT, path)
#     try:
#         with Image.open(full_path) as img:
#             return img.size
#     except Exception:
#         return (np.nan, np.nan)

if __name__ == "__main__":
    combined.to_csv("/work/1mxray/Xray/metadata.csv", index=True)
    print(f"Done. {len(combined)} rows written.")