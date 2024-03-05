import os
import csv
import shutil
import subprocess
import json

def create_mapping(csv_file, mapping_file):
    mappings = {}
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source = row['source_column']
            target = row['target_column']
            mappings[source] = target

    with open(mapping_file, 'w') as outfile:
        for source, target in mappings.items():
            outfile.write(f"<{source}, {target}>\n")

def process_folders(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.startswith("predicted_lineage_") and file_name.endswith(".csv"):
                    csv_file = os.path.join(folder_path, file_name)
                    mapping_file = os.path.join(folder_path, 'mapping.txt')
                    create_mapping(csv_file, mapping_file)
            # Move the folder to Training Data
            destination_folder = os.path.join('Training Data', folder_name)
            shutil.move(folder_path, destination_folder)

    # Run feature_engineering.py
    subprocess.run(['python', 'feature_engineering.py'])

    # Run parameter tuning
    tune_output = subprocess.check_output(['python', 'train.py', '--tune'])
    tune_output_str = tune_output.decode('utf-8')
    best_params = extract_best_params(tune_output_str)

    # Run train.py with best parameters
    subprocess.run(['python', 'train.py', '--params', str(best_params['max_depth']), str(best_params['eta'])])

def extract_best_params(output_str):
    # Parse the output string to extract best parameters
    # Assuming the output is in JSON format
    result = json.loads(output_str)
    best_params = result['best_params']
    return best_params

# Replace 'Test Data' with the path to your root folder
root_folder = 'Test Data'
process_folders(root_folder)
