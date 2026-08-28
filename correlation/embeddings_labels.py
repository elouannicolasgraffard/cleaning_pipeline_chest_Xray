import sys
import os
import torch
import numpy as np
import pandas as pd
import open_clip
import plotly.graph_objects as go
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import umap
from correlation.mergejsontocsv import load_json
import CLIP.config.config as config

class LabeledEvalDataset(Dataset):
    def __init__(self, json_data, img_dir, preprocess, classnames):
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.classnames = classnames
        
        self.samples = []
        for filename, label_array in json_data.items():
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                if isinstance(label_array, list):
                    label_vector = [float(x) for x in label_array]
                    active_labels = [self.classnames[i] for i, val in enumerate(label_array) if val == 1]
                    if not active_labels:
                        active_labels = ["clean_base"]
                else:
                    continue
                
                self.samples.append({
                    'path': img_path,
                    'filename': filename,
                    'label_vector': np.array(label_vector, dtype=np.float32),
                    'active_labels': active_labels
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = self.preprocess(Image.open(sample['path']).convert('RGB'))
        return img, torch.tensor(sample['label_vector']), sample['filename'], ", ".join(sample['active_labels'])

def main():
    if len(sys.argv) < 2:
        print("Usage: python embeddings_labels.py <path_to_labeled_json>")
        sys.exit(1)

    CLASSNAMES = ["collimation", "good_contrast", "medical_device", "truncated", "underexposed_noisy", "rotated", "lateral"]
    images_dir = "/mnt/hpccs01/work/1mxray/CXR8_dataset/images/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Initializing model space on: {device.upper()}")
    model, _, preprocess = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained="")
    checkpoint = torch.load("/home/nicolasg/dev/DualCoOp/output/XRAY-DualCoop-RN101-cosine-bs32-e50/model_best.pth.tar", map_location=device, weights_only=False)
    
    weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=False)
    model = model.to(device).eval()

    raw_json_data = load_json(sys.argv[1])
    dataset = LabeledEvalDataset(raw_json_data, images_dir, preprocess, CLASSNAMES)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    print(f"Loaded {len(dataset)} verified targets.")

    features_list, filenames, annotations_list, binary_labels = [], [], [], []
    
    print("Extracting latent space vectors...")
    with torch.no_grad():
        for imgs, lbl_vecs, fnames, active_strs in dataloader:
            imgs = imgs.to(device)
            img_emb = model.encode_image(imgs)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            
            features_list.append(img_emb.cpu().numpy())
            filenames.extend(fnames)
            annotations_list.extend(active_strs)
            binary_labels.append(lbl_vecs.numpy())

    features = np.concatenate(features_list, axis=0)
    binary_labels = np.concatenate(binary_labels, axis=0)

    print("Running 3D UMAP manifold embedding reduction...")
    n_neighbors = min(15, len(features) - 1)
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, metric='cosine', random_state=42)
    coords_3d = reducer.fit_transform(features)

    # --- SIMPLIFIED COLOR LOGIC ---
    # Good Contrast = Green, Underexposed Noisy = Blue, Everything else = Gray
    point_colors = []
    for vec in binary_labels:
        is_good = vec[1] == 1  # good_contrast
        is_noisy = vec[4] == 1 # underexposed_noisy
        
        if is_good and not is_noisy:
            point_colors.append('#2ca02c') # Solid Green
        elif is_noisy and not is_good:
            point_colors.append('#1f77b4') # Solid Blue
        elif is_good and is_noisy:
            point_colors.append('#ff7f0e') # Orange for rare overlapping edges
        else:
            point_colors.append('#7f7f7f') # Gray for background structural elements

    fig = go.Figure()

    # 1. Plot the Image Bubbles
    fig.add_trace(go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1],
        z=coords_3d[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color=point_colors, # Clear conditional color assignments
            opacity=0.7,
            line=dict(width=1, color='rgb(50,50,50)')
        ),
        text=[f"<b>File:</b> {f}<br><b>Active Flags:</b> {a}" for f, a in zip(filenames, annotations_list)],
        hoverinfo='text',
        name='Images'
    ))

    # 2. Compute and Plot the Mean Label Centroids
    print("Calculating and overlaying mean centers per label...")
    
    # Predefined color assignments for the 7 centroids to stand out sharply
    centroid_colors = ['#e377c2', '#2ca02c', '#ff7f0e', '#9467bd', '#1f77b4', '#bcbd22', '#8c564b']

    for i, classname in enumerate(CLASSNAMES):
        has_label_mask = (binary_labels[:, i] == 1)
        
        if np.sum(has_label_mask) > 0:
            class_coords = coords_3d[has_label_mask]
            mean_center = np.mean(class_coords, axis=0)
            
            fig.add_trace(go.Scatter3d(
                x=[mean_center[0]],
                y=[mean_center[1]],
                z=[mean_center[2]],
                mode='markers+text',
                marker=dict(
                    size=16,
                    color=centroid_colors[i],
                    symbol='diamond',
                    line=dict(width=2, color='white'),
                    opacity=1.0
                ),
                text=[f"<b>CENTER: {classname}</b>"],
                textposition="top center",
                hoverinfo='text',
                name=f"Center: {classname}"
            ))

    fig.update_layout(
        title="DualCoOp 3D Labeled UMAP Space (Green=Good, Blue=Noisy, Gray=Other)",
        scene=dict(
            xaxis_title='UMAP Dim 1',
            yaxis_title='UMAP Dim 2',
            zaxis_title='UMAP Dim 3'
        ),
        margin=dict(r=0, l=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    output_html = "./albeled_data.html"
    fig.write_html(output_html)
    print(f"Interactive model space built successfully! Open file natively: {output_html}")

if __name__ == "__main__":
    main()