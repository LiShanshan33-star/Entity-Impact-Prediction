"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

import os
import sys
import pickle
import gzip
from datetime import datetime, date
import numpy as np
import pandas as pd
import time
import copy

log_folder = "./logs" # log folder
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
    
data_folder=r"./data/output"
data_file=os.path.join(data_folder,'merged_edgess.pkl.gz')   

store_folder="./data/data_concept_graph"
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd) # get parent directory
new_dir_path = os.path.join(parent_dir, store_folder)
os.makedirs(new_dir_path, exist_ok=True)

store_data_file = os.path.join(new_dir_path, "full_dynamic_graphs.parquet")

logsfile=os.path.join(log_folder,"logs_process_pairs.txt")
starting_time=time.time()
print(f'{datetime.now()}: read full graph')
with open(logsfile, "a") as myfile:
    myfile.write(f'\n{datetime.now()}: read full graph') 

with gzip.open(data_file, 'rb') as f: # load the edge list
    full_dynamic_edges = pickle.load(f)

with open(logsfile, "a") as myfile:
    myfile.write(f"\n{datetime.now()}: Done, Total: {len(full_dynamic_edges)}; Elapsed time: {time.time() - starting_time} seconds\n")

#  ... 
print(f"\nData type: {type(full_dynamic_edges)}")
print(f" ... : {type(full_dynamic_edges[0]) if len(full_dynamic_edges) > 0 else 'N/A'}")
print(f" ... : {full_dynamic_edges[0] if len(full_dynamic_edges) > 0 else 'N/A'}")

#  ... 
print(f"\nChecking structure of first few data items:")
for i in range(min(10, len(full_dynamic_edges))):
    item = full_dynamic_edges[i]
    if isinstance(item, tuple):
        print(f"Item {i}: tuple of length {len(item)}, content = {item}")
    elif isinstance(item, list):
        print(f"Item {i}: list of length {len(item)}, content = {item}")
    else:
        print(f"Item {i}: {type(item)}, content = {item}")

#  ... 
# 1:  ... concept pairs ... 
if len(full_dynamic_edges) > 0 and isinstance(full_dynamic_edges[0], (tuple, list)) and len(full_dynamic_edges[0]) == 2:
    print("\nDetected data as concept pairs, will create basic graph structure")
    
    #  ... DataFrame ... 
    full_graph_df = pd.DataFrame(full_dynamic_edges, columns=['concept1', 'concept2'])
    
    #  ... 0
    full_graph_df['time'] = 0
    
    #  ... 
    full_graph_df['ct'] = 0
    
    #  ... 2025-2012
    for year in range(2025, 2011, -1):
        full_graph_df[f'c{year}'] = 0
    
    with open(logsfile, "a") as myfile:
        myfile.write(f"\n ...  {len(full_graph_df)} ")
        myfile.write(f"\n ... ")
    
else:
    # 2:  ... 
    print("\n ... ")
    
    #  ... 
    length_counts = {}
    invalid_items = []
    full_dynamic_edges_copy = []
    
    for i, item in enumerate(full_dynamic_edges):
        item_length = len(item)
        length_counts[item_length] = length_counts.get(item_length, 0) + 1
        
        if item_length < 4:
            with open(logsfile, "a") as myfile:
                myfile.write(f"\nWarning: Item {i} has insufficient length: {item_length}, skipping")
            invalid_items.append(i)
            continue
        
        try:
            if item_length >= 5 and item[4] is not None:
                # citation_per_year
                if isinstance(item[4], list):
                    years_data = {year_data['year']: year_data['cited_by_count'] 
                                 for year_data in item[4] if isinstance(year_data, dict)}
                elif isinstance(item[4], dict):
                    years_data = item[4]
                else:
                    years_data = {}
            else:
                years_data = {}
                
            #  ... 20252012
            new_list = [years_data.get(year, 0) for year in range(2025, 2011, -1)]
            
            #  ... 
            base_item = item[:4] if item_length >= 4 else item + [0] * (4 - item_length)
            full_dynamic_edges_copy.append(base_item + new_list)
            
        except Exception as e:
            with open(logsfile, "a") as myfile:
                myfile.write(f"\nError processing item {i}: {e}, item content: {item}")
            base_item = item[:4] if item_length >= 4 else item + [0] * (4 - item_length)
            full_dynamic_edges_copy.append(base_item + [0] * 14)
            invalid_items.append(i)
        
        if i % 200000 == 0 and i > 0:
            with open(logsfile, "a") as myfile:
                myfile.write(f"\nProcessing item {i+1}/{len(full_dynamic_edges)}")
    
    #  ... 
    with open(logsfile, "a") as myfile:
        myfile.write(f"\n\nProcessing completed!")
        myfile.write(f"\nLength distribution of original items: {length_counts}")
        myfile.write(f"\nNumber of invalid items skipped/fixed: {len(invalid_items)}")
        myfile.write(f"\nFinal processed items count: {len(full_dynamic_edges_copy)}")
    
    print(f"\nLength distribution: {length_counts}")
    print(f"Invalid items: {len(invalid_items)}")
    print(f"Final items: {len(full_dynamic_edges_copy)}")
    
    if len(full_dynamic_edges_copy) > 0:
        time_start = time.time()
        full_graph = np.array(full_dynamic_edges_copy, dtype=object)
        with open(logsfile, "a") as myfile:
            myfile.write(f"\nDone, convert array; Elapsed time: {time.time() - time_start} seconds")
        
        time_start = time.time()
        
        # DataFrame
        columns = ['v1', 'v2', 'time', 'ct'] + [f'c{year}' for year in range(2025, 2011, -1)]
        full_graph_df = pd.DataFrame(full_graph, columns=columns)
        
        #  ... 
        for col in columns[:4]:
            full_graph_df[col] = pd.to_numeric(full_graph_df[col], errors='coerce')
    else:
        print(" ... ")
        full_graph_df = pd.DataFrame()

# DataFrame
if len(full_graph_df) > 0:
    time_start = time.time()
    full_graph_df.to_parquet(store_data_file, compression='gzip')
    
    with open(logsfile, "a") as myfile:
        myfile.write(f"\n{datetime.now()}: Done, full_graph: {len(full_graph_df)}; Elapsed time: {time.time() - time_start} seconds")
    
    print(f"\nDataFrame: {full_graph_df.shape}")
    print(f"DataFrame: {full_graph_df.columns.tolist()}")
    print(f"\n5 ... :")
    print(full_graph_df.head())
else:
    print(" ... ")
    with open(logsfile, "a") as myfile:
        myfile.write(f"\n{datetime.now()}:  -  ... ")

print(f"\n ... ")
