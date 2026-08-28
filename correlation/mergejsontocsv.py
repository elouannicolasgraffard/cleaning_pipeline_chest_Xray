import os
import json
import pandas as pd

LABEL_JSON = "./image_labels.json"
METRIC_JSON = "./metrics_img.json"
OUTPUT_CSV = "./merged_image_data.csv"

# The 7 specific labels matching your architecture array position
CLASSNAMES = [
    "collimation", 
    "good_contrast", 
    "medical_device", 
    "truncated", 
    "underexposed_noisy", # changed slash to underscore for clean CSV columns
    "rotated", 
    "lateral"
]
METRICS = [
         "niqe",
            "brisque",
            "contrast_ratio",
            "entropy",
            "sharpness" ,
            "skewness"
]


def load_json(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {file_path} was empty or corrupted.")
    else:
        print(f"Error: {file_path} does not exist.")
    return {}

def main():
    # 1. Load data
    labels_data = load_json(LABEL_JSON)
    metrics_data = load_json(METRIC_JSON)

    if not labels_data or not metrics_data:
        print("Stopping: One or both JSON structures are empty.")
        return

    # 2. Convert labels JSON to DataFrame and set specific column names
    df_labels = pd.DataFrame.from_dict(labels_data, orient='index', columns=CLASSNAMES)
    
    # 3. CRITICAL STEP: Convert 1s and 0s to True and False
    df_labels = df_labels.astype(bool)
    
    # Clean up the index so 'filename' is a proper column
    df_labels.index.name = 'filename'
    df_labels = df_labels.reset_index()

    # 4. Convert metrics JSON to DataFrame
    df_metrics = pd.DataFrame.from_dict(metrics_data, orient='index', columns = METRICS)
    df_metrics.index.name = 'filename'
    df_metrics = df_metrics.reset_index()

    # 5. Merge dataframes on the common 'filename' column
    merged_df = pd.merge(df_labels, df_metrics, on='filename', how='inner')

    # 6. Save to CSV
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully merged {len(merged_df)} images with boolean labels into {OUTPUT_CSV}!")

if __name__ == '__main__':
    main()