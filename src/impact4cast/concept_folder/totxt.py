"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

import gzip
import pickle

file_path = r"E:\study\research\ASIST\concept_citation\concept_part_150.gz"

# 读取压缩的pickle文件
with gzip.open(file_path, 'rb') as f:
    data = pickle.load(f)

# 查看数据结构
print(f"数据类型: {type(data)}")
print(f"数据内容: {data}")