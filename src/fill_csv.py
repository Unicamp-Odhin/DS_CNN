import os
import csv

# Configuration
folder_path = 'mfccs/train/keyword'  # Change this
target_lines = 399  # Total lines including header

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', newline='', encoding='utf-8') as infile:
            reader = list(csv.reader(infile))
            header = reader[0]
            data = reader[1:]

        current_lines = len(data)
        num_cols = len(header)

        # Add zero rows until we have 399 data lines
        while current_lines < (target_lines - 1):
            data.append([0] * num_cols)
            current_lines += 1

        # Write back to the same file
        with open(file_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(data)

        print(f"Padded: {filename} to {len(data) + 1} lines")
