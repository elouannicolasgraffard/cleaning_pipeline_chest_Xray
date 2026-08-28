import os
import json
import shutil
import sys

# Target directory where DualCoOp expects images
DEST_IMAGES_DIR = "/mnt/hpccs01/work/1mxray/dual_coop_dataset/images/"

# Exact order of classes required by your dataset
CLASSNAMES = [
    "collimation", 
    "good_contrast", 
    "medical_device", 
    "truncated", 
    "underexposed/noisy", 
    "rotated", 
    "lateral"
]

def append_to_dataset(raw_json_path, target_json_path):
    if not os.path.exists(raw_json_path):
        print(f"❌ Error: New incoming file '{raw_json_path}' not found.")
        return

    os.makedirs(DEST_IMAGES_DIR, exist_ok=True)

    # 1. Load existing target JSON if it exists, otherwise start fresh
    if os.path.exists(target_json_path):
        with open(target_json_path, "r") as f:
            dataset_dict = json.load(f)
        print(f" Loaded existing dataset with {len(dataset_dict)} items from '{target_json_path}'")
    else:
        dataset_dict = {}
        print(f"🆕 Target file '{target_json_path}' doesn't exist yet. Creating new file.")

    # 2. Load incoming new data
    with open(raw_json_path, "r") as f:
        new_raw_data = json.load(f)

    added_count = 0
    copied_count = 0
    missing_count = 0

    print(f" Processing {len(new_raw_data)} new/updated entries...\n")

    for full_path, label_list in new_raw_data.items():
        img_name = full_path

        # Convert string list to 0/1 binary vector
        binary_vector = [1 if cls in label_list else 0 for cls in CLASSNAMES]

        # Add or update entry in existing dataset dictionary
        dataset_dict[img_name] = binary_vector
        added_count += 1

        # Copy physical image file
        if os.path.exists(full_path):
            dest_path = os.path.join(DEST_IMAGES_DIR, img_name)
            dest_path = img_name
            try:
                shutil.copy(full_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Failed to copy {img_name}: {e}")
        else:
            print(f" Warning: Image file missing at '{full_path}'")
            missing_count += 1

    # 3. Save the merged dictionary back to the target JSON
    target_dir = os.path.dirname(target_json_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    with open(target_json_path, "w") as f:
        json.dump(dataset_dict, f, indent=4)

    print(f" Successfully appended new entries!")
    print(f" └─ Total images in '{target_json_path}' now: {len(dataset_dict)}")
    print(f" └─ Copied {copied_count} new images to: {DEST_IMAGES_DIR}")
    if missing_count > 0:
        print(f" └─ ⚠️ {missing_count} source images were not found on disk.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python append_dataset.py <new_labels.json> <target_train.json>")
        print("Example: python append_dataset.py new_batch.json /mnt/hpccs01/work/1mxray/dual_coop_dataset/labels/train.json")
        sys.exit(1)

    input_json = sys.argv[1]
    target_json = sys.argv[2]
    
    append_to_dataset(input_json, target_json)