import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """Lightweight dataset wrapper for pre-computed feature arrays."""
    def __init__(self, features, labels, paths):
        self.features = torch.tensor(features, dtype=torch.float32) if isinstance(features, np.ndarray) else features
        self.labels = torch.tensor(labels, dtype=torch.float32) if isinstance(labels, np.ndarray) else labels
        self.paths = paths

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.paths[idx]


def load_dataset_by_source(metadata_path, npy_features_path, target_sources=None):
    """
    Loads metadata and features from a single CSV and NPY file,
    filtering strictly by the 'source' column.
    
    Parameters:
    - metadata_path: path to your unified master CSV file.
    - npy_features_path: path to your unified .npy feature array.
    - target_sources: list of dataset source names to keep 
                      (e.g., ['CheXpert', 'MIMIC'] for training, or ['CXR8'] for inference).
                      If None, loads everything.
    """
    df = pd.read_csv(metadata_path, low_memory=False)
    features_all = np.load(npy_features_path)
    
    # Safety check: CSV rows must align 1:1 with NPY features array
    if len(df) != len(features_all):
        raise ValueError(f"❌ Row count mismatch! CSV has {len(df):,} rows, but NPY has {len(features_all):,} items.")

    img_col = next((c for c in ['path', 'Path', 'file_path'] if c in df.columns), df.columns[1])
    view_col = next((c for c in ['frontal/lateral'] if c in df.columns), None)
    source_col = next((c for c in ['source', 'Source', 'dataset'] if c in df.columns), None)

    if not source_col:
        raise ValueError("❌ Could not find a 'source' column in your CSV file to filter by dataset.")

    paths, labels, selected_indices = [], [], []

    for idx, row in df.iterrows():
        src = str(row[source_col]).strip()
        
        # Check if this row belongs to one of the target sources
        match_found = False
        if target_sources:
            for t_src in target_sources:
                if t_src.lower() in src.lower():
                    match_found = True
                    break
        else:
            match_found = True  # Load everything if no filter specified

        if match_found:
            fname = str(row[img_col])
            is_lat = 0
            if view_col and pd.notna(row[view_col]):
                val = str(row[view_col]).lower()
                is_lat = 1 if ('lateral' in val or val == '1') else 0
            
            paths.append(fname)
            labels.append(is_lat)
            selected_indices.append(idx)

    # Subset the large feature array using matching indices to maintain perfect alignment
    features_filtered = features_all[selected_indices]
    labels_arr = np.array(labels, dtype=np.float32)

    print(f"📖 Loaded {len(paths):,} records matching sources {target_sources} from '{metadata_path}'")
    return features_filtered, labels_arr, paths