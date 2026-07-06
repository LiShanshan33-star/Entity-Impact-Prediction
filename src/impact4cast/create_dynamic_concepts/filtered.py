"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

import os
import gzip
import pickle

# === 设置你的文件夹路径 ===
folder_path = r"E:\study\research\ASIST\concept_citation"

# 输出路径（保存筛选后的结果）
output_folder = os.path.join(folder_path, "filtered")
os.makedirs(output_folder, exist_ok=True)

# 获取所有 .gz 文件
gz_files = [f for f in os.listdir(folder_path) if f.endswith(".gz")]
print(f"共找到 {len(gz_files)} 个文件。")

total_before = 0
total_after = 0

# 循环处理每个文件
for gz_file in gz_files:
    file_path = os.path.join(folder_path, gz_file)
    output_path = os.path.join(output_folder, gz_file)

    try:
        with gzip.open(file_path, "rb") as f:
            edge_list = pickle.load(f)
    except Exception as e:
        print(f" 无法读取 {gz_file}: {e}")
        continue

    total_before += len(edge_list)

    # 筛选：共被引次数不为 0
    filtered = [
        edge for edge in edge_list
        if len(edge) > 3 and isinstance(edge[3], (int, float)) and edge[3] != 0
    ]

    total_after += len(filtered)

    # 打印前几条看看
    print(f"\n {gz_file}")
    print(f"原始记录: {len(edge_list)}  →  筛选后: {len(filtered)}")
    if filtered:
        for e in filtered[:5]:
            print(e)

        # 保存筛选结果
        with gzip.open(output_path, "wb") as f:
            pickle.dump(filtered, f)
    else:
        print("无符合条件的记录，跳过保存。")

# print("\n=== 全部完成 ===")
# print(f"总记录数: {total_before} → 保留 {total_after} （{total_after/total_before*100:.2f}%）")
# print(f"筛选后文件保存在: {output_folder}")