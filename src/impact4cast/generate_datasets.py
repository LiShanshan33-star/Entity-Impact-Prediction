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
    生成所有需要的训练和评估数据集
    """
    
    # 1. 加载基础数据
    print("Loading full dynamic graph...")
    full_graph = pd.read_parquet(full_graph_path)
    concepts = load_concept_list(concept_list_path)
    
    # 2. 生成所有可能的概念对
    all_pairs = list(combinations(concepts, 2))
    
    # 3. 为不同的时间窗口生成训练数据
    train_years = [
        (2016, 2019),  # 训练窗口1
        (2017, 2020),  # 训练窗口2
        (2018, 2021),  # 训练窗口3
    ]
    
    for y1, y2 in train_years:
        for IR in [10, 50]:  # 不同的影响力阈值
            print(f"Generating train data: {y1}-{y2}, IR={IR}")
            
            # 生成训练数据
            train_file = generate_train_data(
                y1, y2, IR, full_graph, concepts, all_pairs, feature_folder
            )
            
            # 移动到目标文件夹
            os.rename(train_file, 
                     os.path.join(output_folder, f'train_data_{y1}_{y2}_IR{IR}.parquet'))
            
            gc.collect()
    
    # 4. 生成评估数据
    eval_years = [
        (2020, 2022),
        (2019, 2022),  # 对应代码中的设置
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
    生成单个训练数据文件
    """
    # 获取y1年前已有的边
    edges_before_y1 = full_graph[full_graph['time'] <= f'{y1}-12-31']
    existing_pairs = set(zip(edges_before_y1['v1'], edges_before_y1['v2']))
    
    # 找出未连接的pair
    unconnected = []
    for v1, v2 in all_pairs:
        if (v1, v2) not in existing_pairs and (v2, v1) not in existing_pairs:
            unconnected.append((v1, v2))
    
    # 分批处理避免OOM
    batch_size = 100000
    all_data = []
    
    for i in range(0, len(unconnected), batch_size):
        batch_pairs = unconnected[i:i+batch_size]
        batch_data = []
        
        for v1, v2 in batch_pairs:
            # 获取标签
            edges_after = full_graph[
                (full_graph['time'] <= f'{y2}-12-31') &
                (((full_graph['v1'] == v1) & (full_graph['v2'] == v2)) |
                 ((full_graph['v1'] == v2) & (full_graph['v2'] == v1)))
            ]
            
            citations_y2 = edges_after[f'c{y2}'].sum() if len(edges_after) > 0 else 0
            label = 1 if citations_y2 >= IR else 0
            
            # 提取特征（这里需要实现具体的特征提取逻辑）
            features = extract_features(v1, v2, y1, feature_folder)
            
            batch_data.append([v1, v2, label] + features)
        
        all_data.extend(batch_data)
        
        # 定期保存中间结果
        if (i // batch_size) % 10 == 0:
            temp_df = pd.DataFrame(all_data)
            temp_df.to_parquet(f'temp_train_{y1}_{y2}_IR{IR}_part_{i}.parquet')
            all_data = []  # 清空内存
    
    # 合并所有批次
    final_df = pd.concat([
        pd.read_parquet(f) for f in os.listdir() if f.startswith(f'temp_train_{y1}_{y2}_IR{IR}')
    ])
    
    # 删除临时文件
    for f in os.listdir():
        if f.startswith(f'temp_train_{y1}_{y2}_IR{IR}'):
            os.remove(f)
    
    # 保存最终文件
    output_file = f'train_data_{y1}_{y2}_IR{IR}.parquet'
    final_df.to_parquet(output_file)
    
    return output_file

if __name__ == '__main__':
    generate_all_datasets()