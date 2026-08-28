import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Your base dataset directory (leave empty if paths in .txt are already complete/absolute)
DATASET_DIR = ""


def load_and_normalize_image(img_path):
    """
    Handles 16-bit medical images (PadChest, DICOM, 16-bit PNG/TIFF) by scaling
    pixel values to the standard 8-bit range (0-255) for rendering.
    """
    raw_img = Image.open(img_path)
    img_np = np.array(raw_img).astype(np.float32)

    # Calculate min and max pixel intensity
    img_min = img_np.min()
    img_max = img_np.max()

    # Perform min-max scaling to stretch dynamic range to 0-255
    if img_max > img_min:
        img_normalized = (img_np - img_min) / (img_max - img_min) * 255.0
    else:
        img_normalized = np.zeros_like(img_np)

    # Convert to 8-bit integer and convert to RGB image
    img_uint8 = img_normalized.astype(np.uint8)
    return Image.fromarray(img_uint8).convert('RGB')


def plot_image_grid(image_paths, output_filename="visualized_grid.png", max_images=200, cols=5):
    """
    Loads up to `max_images` from `image_paths` and saves them in a single grid layout.
    """
    # 1. Filter out files that don't exist on disk
    valid_paths = [p for p in image_paths if os.path.exists(p)]

    if not valid_paths:
        print("❌ Error: None of the image paths in the file exist on disk!")
        return

    # Cap at max_images (200)
    valid_paths = valid_paths[:max_images]
    n = len(valid_paths)

    # 2. Calculate grid dimensions (200 images / 5 cols = 40 rows)
    rows = (n + cols - 1) // cols
    print(f"📊 Found {n} valid images. Normalizing 16-bit pixel values and rendering grid ({rows} rows x {cols} cols)...")

    # Proportionate figure size so 40 rows render cleanly
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.2))

    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 3. Load, normalize, and place each image in the subplot grid
    for i, img_path in enumerate(valid_paths):
        try:
            img = load_and_normalize_image(img_path)
        except Exception as e:
            print(f"⚠️ Warning: Could not open {img_path}: {e}")
            img = Image.new('RGB', (518, 518), color='gray')

        axes[i].imshow(img)
        # Display the file name as subplot title
        axes[i].set_title(f"#{i+1}: {os.path.basename(img_path)}", fontsize=8, fontweight="bold")
        axes[i].axis('off')

    # 4. Turn off unused subplot axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    # 5. Save single grid image
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=200)
    plt.close('all')
    print(f"💾 Saved complete image grid to '{output_filename}'")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_grid.py <path_to_image_list.txt>")
        sys.exit(1)

    image_list_file = sys.argv[1]

    if not os.path.exists(image_list_file):
        print(f"Error: The file '{image_list_file}' does not exist.")
        sys.exit(1)

    print(f"Reading image list from: {image_list_file}")

    image_paths = []
    with open(image_list_file, 'r') as f:
        for line in f:
            img_name = line.strip()
            if not img_name:
                continue

            img_path = os.path.join(DATASET_DIR, img_name) if DATASET_DIR else img_name
            image_paths.append(img_path)

    plot_image_grid(image_paths, output_filename="visualize.png", max_images=500, cols=8)


if __name__ == "__main__":
    main()