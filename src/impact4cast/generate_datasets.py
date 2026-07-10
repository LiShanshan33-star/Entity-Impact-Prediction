"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

# generate_datasets.py
import os
import pandas as pd
import numpy as np
from itertools import combinations
import pyarrow.parquet as pq
import gc

def generate_all_datasets(
    full_graph_path='full_dynamic_graph.parquet',
    concept_list_path='concept_list.txt',
    feature_folder='data_for_features',
    output_folder='data_pair_solution',
    eval_output_folder='data_eval'
):
    """
    Generate all required training and evaluation datasets.
    """
    
    # 1. Load base data
    print("Loading full dynamic graph...")
    full_graph = pd.read_parquet(full_graph_path)
    concepts = load_concept_list(concept_list_path)
    
    # 2. Generate all possible concept pairs
    all_pairs = list(combinations(concepts, 2))
    
    # 3. Generate training data for different time windows
    train_years = [
        (2016, 2019),  #  ... 1
        (2017, 2020),  #  ... 2
        (2018, 2021),  #  ... 3
    ]
    
    for y1, y2 in train_years:
        for IR in [10, 50]:  #  ... 
            print(f"Generating train data: {y1}-{y2}, IR={IR}")
            
            #  ... 
            train_file = generate_train_data(
                y1, y2, IR, full_graph, concepts, all_pairs, feature_folder
            )
            
            #  ... 
            os.rename(train_file, 
                     os.path.join(output_folder, f'train_data_{y1}_{y2}_IR{IR}.parquet'))
            
            gc.collect()
    
    # 4. Generate evaluation data
    eval_years = [
        (2020, 2022),
        (2019, 2022),  #  ... 
    ]
    
    for y1_eval, y2_eval in eval_years:
        for IR in [10, 50]:
            print(f"Generating eval data: {y1_eval}-{y2_eval}, IR={IR}")
            
            eval_file = generate_eval_data(
                y1_eval, y2_eval, IR, full_graph, concepts, all_pairs, feature_folder
            )
            
            os.rename(eval_file,
                     os.path.join(eval_output_folder, f'eval_data_{y1_eval}_{y2_eval}_IR{IR}.parquet'))
            
            gc.collect()

def generate_train_data(y1, y2, IR, full_graph, concepts, all_pairs, feature_folder):
    """
     ... 
    """
    # y1 ... 
    edges_before_y1 = full_graph[full_graph['time'] <= f'{y1}-12-31']
    existing_pairs = set(zip(edges_before_y1['v1'], edges_before_y1['v2']))
    
    #  ... pair
    unconnected = []
    for v1, v2 in all_pairs:
        if (v1, v2) not in existing_pairs and (v2, v1) not in existing_pairs:
            unconnected.append((v1, v2))
    
    #  ... OOM
    batch_size = 100000
    all_data = []
    
    for i in range(0, len(unconnected), batch_size):
        batch_pairs = unconnected[i:i+batch_size]
        batch_data = []
        
        for v1, v2 in batch_pairs:
            #  ... 
            edges_after = full_graph[
                (full_graph['time'] <= f'{y2}-12-31') &
                (((full_graph['v1'] == v1) & (full_graph['v2'] == v2)) |
                 ((full_graph['v1'] == v2) & (full_graph['v2'] == v1)))
            ]
            
            citations_y2 = edges_after[f'c{y2}'].sum() if len(edges_after) > 0 else 0
            label = 1 if citations_y2 >= IR else 0
            
            #  ... 
            features = extract_features(v1, v2, y1, feature_folder)
            
            batch_data.append([v1, v2, label] + features)
        
        all_data.extend(batch_data)
        
        #  ... 
        if (i // batch_size) % 10 == 0:
            temp_df = pd.DataFrame(all_data)
            temp_df.to_parquet(f'temp_train_{y1}_{y2}_IR{IR}_part_{i}.parquet')
            all_data = []  #  ... 
    
    # Merge ... 
    final_df = pd.concat([
        pd.read_parquet(f) for f in os.listdir() if f.startswith(f'temp_train_{y1}_{y2}_IR{IR}')
    ])
    
    #  ... 
    for f in os.listdir():
        if f.startswith(f'temp_train_{y1}_{y2}_IR{IR}'):
            os.remove(f)
    
    #  ... 
    output_file = f'train_data_{y1}_{y2}_IR{IR}.parquet'
    final_df.to_parquet(output_file)
    
    return output_file

if __name__ == '__main__':
    generate_all_datasets()