import os
import csv

# Configuration
folder_path = 'DS_CNN/mfccs/train/background'  # Change this to your folder
max_lines = 399  # Includes the header line

# Ensure the output doesn't overwrite the original
output_suffix = "_trimmed"

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}{output_suffix}.csv")

        with open(input_path, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            rows = []
            for i, row in enumerate(reader):
                if i >= max_lines:
                    break
                rows.append(row)

        with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(rows)

        print(f"Trimmed: {filename} → {os.path.basename(output_path)}")
