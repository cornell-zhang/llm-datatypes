import re
import sys
import glob
import os

def parse_acc_values(file_path):
    # Pattern to match acc values directly
    pattern = re.compile(r'\|\s*(?:[\w_]+\s*\|\s*\d+\s*\|)?acc\s*\|\s*([\d.]+)\s*\|')
    acc_values = []

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                acc_values.append(match.group(1))

    return acc_values

def main(input_directory, output_file):
    # Find all *.out files in the input directory and sort them alphabetically
    input_files = sorted(glob.glob(os.path.join(input_directory, '*.out')))
    
    with open(output_file, 'w') as out_file:
        for file_path in input_files:
            acc_values = parse_acc_values(file_path)
            # Include the filename (without the path) before the values
            file_name = os.path.basename(file_path)
            out_file.write(file_name + '\n' + ','.join(acc_values) + '\n\n')

# Example usage: python script.py /path/to/input/directory /path/to/output.csv
if __name__ == "__main__":
    input_directory_path = sys.argv[1]  # First argument is the input directory path
    output_file_path = sys.argv[2]      # Second argument is the output file path

    main(input_directory_path, output_file_path)
