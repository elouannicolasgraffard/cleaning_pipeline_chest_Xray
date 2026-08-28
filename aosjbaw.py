from pathlib import Path

# Define your paths (update these to your actual directory paths on the cluster)
# 'source_dir' is where the folders with the .txt files are (downstairs)
# 'dest_dir' is where your image folders are (upstairs)
SOURCE_DIR = Path("/work/1mxray/Xray/files")
DEST_DIR = Path("/work/1mxray/Xray/datasets/MIMIC/physionet.org/files/mimic-cxr-jpg/2.1.0/files")

def move_txt_files():
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory does not exist: {SOURCE_DIR}")
        return

    print(f"🔍 Searching for .txt files in {SOURCE_DIR}...")
    txt_files = list(SOURCE_DIR.rglob("*.txt"))
    print(f"📄 Found {len(txt_files):,} text files to move.")

    moved_count = 0
    for src_path in txt_files:
        # Get the relative path from the source root to maintain folder structure (e.g., p10/12345/report.txt)
        rel_path = src_path.relative_to(SOURCE_DIR)
        
        # Construct the target path upstairs
        dest_path = DEST_DIR / rel_path
        
        # Ensure the destination subfolder exists upstairs
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the file (change to shutil.copy if you want to keep a backup downstairs)
        src_path.rename(dest_path)
        moved_count += 1

    print(f"✅ Successfully moved {moved_count:,} text files to their corresponding folders upstairs!")

if __name__ == "__main__":
    move_txt_files()