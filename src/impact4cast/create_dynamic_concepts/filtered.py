"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

import os
import gzip
import pickle

# ===  ...  ===
folder_path = r"./data/concept_citation"

#  ... 
output_folder = os.path.join(folder_path, "filtered")
os.makedirs(output_folder, exist_ok=True)

#  ...  .gz 
gz_files = [f for f in os.listdir(folder_path) if f.endswith(".gz")]
print(f" ...  {len(gz_files)}  ... ")

total_before = 0
total_after = 0

#  ... 
for gz_file in gz_files:
    file_path = os.path.join(folder_path, gz_file)
    output_path = os.path.join(output_folder, gz_file)

    try:
        with gzip.open(file_path, "rb") as f:
            edge_list = pickle.load(f)
    except Exception as e:
        print(f"  ...  {gz_file}: {e}")
        continue

    total_before += len(edge_list)

    #  ...  0
    filtered = [
        edge for edge in edge_list
        if len(edge) > 3 and isinstance(edge[3], (int, float)) and edge[3] != 0
    ]

    total_after += len(filtered)

    #  ... 
    print(f"\n {gz_file}")
    print(f" ... : {len(edge_list)}  →   ... : {len(filtered)}")
    if filtered:
        for e in filtered[:5]:
            print(e)

        #  ... 
        with gzip.open(output_path, "wb") as f:
            pickle.dump(filtered, f)
    else:
        print(" ... ")

# print("\n===  ...  ===")
# print(f"Record count: {total_before} →  {total_after} {total_after/total_before*100:.2f}%")
# print(f" ... : {output_folder}")