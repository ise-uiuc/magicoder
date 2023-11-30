import argparse
import json
import os
from pathlib import Path

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add the arguments
parser.add_argument(
    "Directory", metavar="dir", type=str, help="the directory to process"
)

# Parse the arguments
args = parser.parse_args()

# Use the directory argument
directory = args.Directory

json_files = list(Path(directory).glob("*.json"))
all_data = [json.loads(file.read_text()) for file in json_files]


def get_language(name: str):
    return name.split("-")[1].split("_")[0]


data_count = {"cpp": 161, "php": 161, "rs": 156, "swift": 158, "js": 161, "java": 158}

sum_pass = 0
total_count = 0
for file, data in zip(json_files, all_data):
    language = get_language(file.name)
    data_number = data_count[language]
    pass_1 = data[f"multiple-{language}"]["pass@1"]
    print("LANGUAGE:", language, "PASS@1:", pass_1)
    sum_pass += pass_1 * data_number
    total_count += data_number
print("AVERAGE:", sum_pass / total_count)
