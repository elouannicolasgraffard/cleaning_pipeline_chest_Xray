# ========================================================
# CRITICAL: THIS MUST GO AT THE ABSOLUTE TOP OF THE FILE
# ========================================================
import cv2
import numpy as np
import scipy.misc
import skimage.transform

# FIX 1: Overcome the modern SciPy deprecation bug
def patched_imresize(arr, size, interp='bicubic', mode=None):
    if isinstance(size, float):
        height, width = arr.shape[:2]
        new_size = (int(width * size), int(height * size))
    else:
        new_size = size
    return cv2.resize(arr, new_size, interpolation=cv2.INTER_CUBIC)

scipy.misc.imresize = patched_imresize

# FIX 2: Overcome the modern NumPy 'np.int' removal bug
if not hasattr(np, "int"):
    setattr(np, "int", int)

# FIX 3: Overcome the BRISQUE / scikit-image 'multichannel' deprecation error
_original_rescale = skimage.transform.rescale
def patched_rescale(*args, **kwargs):
    if 'multichannel' in kwargs:
        is_multichannel = kwargs.pop('multichannel')
        if is_multichannel and 'channel_axis' not in kwargs:
            kwargs['channel_axis'] = -1  # Modern equivalent of multichannel=True
    return _original_rescale(*args, **kwargs)

skimage.transform.rescale = patched_rescale
# ========================================================

import sys
import json
import os
from scipy.stats import skew

# Quality metrics imports
from imquality import brisque
import skvideo
import skvideo.measure
from skimage import io, img_as_float
from skimage.measure import shannon_entropy
from brisque import BRISQUE

OUTPUT_JSON_PATH = "./metrics_img.json"

def calculate_metrics(img_path):
    try:
        image = io.imread(img_path, as_gray=True)
        image2 = cv2.imread(img_path)

        img_input = img_as_float(image)
    
        niqe_score = skvideo.measure.niqe(img_input)
        niqe_val = float(niqe_score[0])
        
        # brisque_val = float(brisque.score(img_input))
        brisque_obj = BRISQUE()
        brisque = brisque_obj.score(image2)
        
        contrast_ratio = float(img_input.std())
        entropy_val = float(shannon_entropy(img_input))
        sharpness_val = float(cv2.Laplacian(img_input, cv2.CV_64F).var())
        skewness_val = float(skew(img_input.ravel()))

        return {
            "niqe": niqe_val,
            "brisque": brisque,
            "contrast_ratio": contrast_ratio,
            "entropy": entropy_val,
            "sharpness": sharpness_val,
            "skewness": skewness_val
        }

    except Exception as e:
        print(f"⚠️ Skipping {os.path.basename(img_path)} due to error: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <text_file.txt>")
        sys.exit(1)

    res = {}
    text_file_path = sys.argv[1]

    if not os.path.exists(text_file_path):
        print(f"Error: Text file '{text_file_path}' not found.")
        sys.exit(1)

    print(f"📖 Reading image paths from {text_file_path}...")
    
    with open(text_file_path, 'r') as f:
        for line in f:
            img_name = line.strip()
            if not img_name:
                continue

            if not os.path.exists(img_name):
                print(f"❌ File not found: {img_name}")
                continue

            print(f"📸 Processing: {os.path.basename(img_name)}")
            metrics = calculate_metrics(img_name)
            
            if metrics is not None:
                res[os.path.basename(img_name)] = metrics
            
    print(f"\nSaving all metrics to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(res, f, indent=4)
        
    print("✅ Done!")

if __name__ == "__main__":
    main()