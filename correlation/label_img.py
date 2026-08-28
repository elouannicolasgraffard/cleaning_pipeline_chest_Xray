import sys
import os
import json

# Output JSON file path in your workspace
OUTPUT_JSON = "image_labels.json"

def main():
    # Ensure a text file path and a label were passed as arguments to the script
    if len(sys.argv) < 3:
        print("Usage: python open_images.py <path_to_image_list.txt> <label>")
        sys.exit(1)

    txt_file_path = sys.argv[1]
    label = sys.argv[2]

    # 1. Load existing data if the JSON file already exists
    data = {}
    if os.path.exists(OUTPUT_JSON):
        print(f"Loading existing JSON entries from: {OUTPUT_JSON}")
        try:
            with open(OUTPUT_JSON, 'r') as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError:
            print("Warning: JSON file was empty or corrupted. Starting fresh.")
            data = {}

    # Check if the input text file exists
    if not os.path.exists(txt_file_path):
        print(f"Error: The input file '{txt_file_path}' does not exist.")
        sys.exit(1)

    print(f"Processing lines from: {txt_file_path} with label: '{label}'")
    
    # 2. Parse the text file and append the label cleanly
    with open(txt_file_path, 'r') as f:
        for line in f:
            img_name = line.strip()
            if not img_name:
                continue  # Skip blank lines

            # If the image string already exists in our dictionary
            if img_name in data:
                # Ensure the value format is a list
                if not isinstance(data[img_name], list):
                    data[img_name] = [data[img_name]]
                
                # Append the label only if it isn't already assigned
                if label not in data[img_name]:
                    data[img_name].append(label)
            else:
                # First time seeing this image string: initialize with a list containing the label
                data[img_name] = [label]

    # 3. Save the updated dictionary back to the JSON file
    print(f"Saving updated JSON map back to: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("Done! Check your updated 'image_labels.json' file.")

if __name__ == "__main__":
    main()