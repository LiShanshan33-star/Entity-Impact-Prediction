"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

import glob
import gzip
import pickle
import os
import time
from datetime import datetime
import sys

# 配置路径
log_folder = 'logs'
vertex_list_folder = 'concept_citation'
full_vertex_lists = os.path.join(vertex_list_folder, 'all_concept_citation.gz')
temp_folder = os.path.join(vertex_list_folder, 'temp_merge')

# 创建必要的文件夹
os.makedirs(log_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

log_files = 'log_merge_concept_citation.txt'

# 使用UTF-8编码打开日志文件
log_file_path = os.path.join(log_folder, log_files)

# 记录开始时间
with open(log_file_path, 'a', encoding='utf-8') as f:
    f.write(f'\n{"="*60}\n')
    f.write(f'开始合并文件 (方案2: 临时文件分批处理): {datetime.now()}\n')
    f.write(f'{"="*60}\n')

# 获取所有需要合并的文件（排除已经合并的文件）
list_file_names = os.listdir(vertex_list_folder)
vertex_file_name_unsorted = [file for file in list_file_names 
                            if file.endswith('.gz') and file != 'all_concept_citation.gz']
vertex_lists_files = sorted(vertex_file_name_unsorted)

print(f"找到 {len(vertex_lists_files)} 个文件需要合并")
print(f"临时文件夹: {temp_folder}")
print(f"批次大小: 500,000 条记录/批")

# ==================== 方案2：分批处理 + 临时文件 ====================
def merge_files_with_temp():
    """使用临时文件分批处理，最后合并（内存最安全）"""
    temp_files = []
    batch_size = 500000  # 每批50万条记录，可根据内存调整
    current_batch = []
    batch_index = 0
    total_records = 0
    empty_count = 0
    
    print("\n第一阶段：读取源文件并创建临时批次...")
    print("-" * 50)
    
    for id_file, curr_vertex_file in enumerate(vertex_lists_files):
        file_path = os.path.join(vertex_list_folder, curr_vertex_file)
        
        try:
            # 读取源文件
            with gzip.open(file_path, 'rb') as f:
                vertex_data_list = pickle.load(f)
            
            file_record_count = len(vertex_data_list) if vertex_data_list else 0
            
            if vertex_data_list:
                # 添加到当前批次
                current_batch.extend(vertex_data_list)
                total_records += file_record_count
                
                # 当批次达到大小时，保存到临时文件
                while len(current_batch) >= batch_size:
                    temp_file = os.path.join(temp_folder, f'temp_batch_{batch_index:04d}.pkl')
                    
                    # 取出前batch_size条记录
                    batch_to_save = current_batch[:batch_size]
                    current_batch = current_batch[batch_size:]
                    
                    # 保存到临时文件
                    with open(temp_file, 'wb') as temp_f:
                        pickle.dump(batch_to_save, temp_f)
                    
                    temp_files.append(temp_file)
                    batch_index += 1
                    
                    print(f"  创建临时文件: {os.path.basename(temp_file)} "
                          f"({len(batch_to_save):,} 条记录)")
            else:
                empty_count += 1
                print(f'  空文件: {curr_vertex_file}')
            
            # 显示进度
            progress = (id_file + 1) / len(vertex_lists_files) * 100
            print(f'处理文件 [{id_file+1}/{len(vertex_lists_files)}] {progress:.1f}%: '
                  f'{curr_vertex_file} ({file_record_count:,} 条) -> '
                  f'累计 {total_records:,} 条记录')
            
            # 记录日志（使用UTF-8编码）
            with open(log_file_path, 'a', encoding='utf-8') as log_f:
                log_f.write(f'处理文件: {curr_vertex_file}; '
                           f'记录数: {file_record_count:,}; '
                           f'累计: {total_records:,}; '
                           f'进度: {progress:.1f}%; '
                           f'临时文件数: {batch_index}\n')
                
        except Exception as e:
            error_msg = f"处理文件 {curr_vertex_file} 时出错: {e}"
            print(f"  ❌ {error_msg}")
            with open(log_file_path, 'a', encoding='utf-8') as log_f:
                log_f.write(f'错误: {error_msg}\n')
    
    # 保存最后一批数据
    if current_batch:
        temp_file = os.path.join(temp_folder, f'temp_batch_{batch_index:04d}_final.pkl')
        with open(temp_file, 'wb') as temp_f:
            pickle.dump(current_batch, temp_f)
        temp_files.append(temp_file)
        print(f"\n创建最后临时文件: {os.path.basename(temp_file)} "
              f"({len(current_batch):,} 条记录)")
        batch_index += 1
    
    print(f"\n第一阶段完成：创建了 {len(temp_files)} 个临时文件，总计 {total_records:,} 条记录")
    
    # 第二阶段：合并所有临时文件
    print("\n第二阶段：合并临时文件到最终文件...")
    print("-" * 50)
    
    merged_count = 0
    with gzip.open(full_vertex_lists, 'wb') as out_f:
        # 对临时文件排序，确保顺序正确
        for temp_file in sorted(temp_files):
            try:
                # 读取临时文件
                with open(temp_file, 'rb') as in_f:
                    batch_data = pickle.load(in_f)
                
                # 写入最终文件
                pickle.dump(batch_data, out_f)
                out_f.flush()
                
                merged_count += len(batch_data)
                print(f"  合并: {os.path.basename(temp_file)} "
                      f"({len(batch_data):,} 条, 累计 {merged_count:,} 条)")
                
                # 删除临时文件
                os.remove(temp_file)
                
            except Exception as e:
                print(f"  ❌ 合并临时文件 {temp_file} 时出错: {e}")
    
    # 删除临时文件夹
    try:
        os.rmdir(temp_folder)
        print(f"\n已删除临时文件夹: {temp_folder}")
    except Exception as e:
        print(f"\n临时文件夹非空，保留: {temp_folder}")
    
    return total_records, empty_count, merged_count


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("="*60)
    print("合并概念引用文件 (方案2: 临时文件分批处理)")
    print("="*60)
    
    # 检查是否有文件需要合并
    if not vertex_lists_files:
        print("没有找到需要合并的文件！")
        sys.exit(1)
    
    
    
    # 开始合并
    start_time = time.time()
    total_records, empty_count, merged_count = merge_files_with_temp()
    elapsed_time = time.time() - start_time
    
    # 验证结果（使用普通字符代替特殊符号）
    if total_records == merged_count:
        verification = "通过 (记录数一致)"
    else:
        verification = f"不匹配 (相差 {abs(total_records - merged_count):,} 条)"
    
    # 记录完成信息（使用UTF-8编码）
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f'\n{"="*60}\n')
        f.write(f'合并完成: {datetime.now()}\n')
        f.write(f'源文件数: {len(vertex_lists_files)}\n')
        f.write(f'临时文件数: {len(glob.glob(os.path.join(temp_folder, "temp_batch_*.pkl")))} 个\n')
        f.write(f'总记录数: {total_records:,}\n')
        f.write(f'空文件数: {empty_count}\n')
        f.write(f'合并记录数: {merged_count:,}\n')
        f.write(f'验证结果: {verification}\n')
        f.write(f'总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)\n')
        f.write(f'输出文件: {full_vertex_lists}\n')
        f.write(f'{"="*60}\n')
    
    print("\n" + "="*60)
    print("合并完成！")
    print("="*60)
    print(f"源文件数: {len(vertex_lists_files)}")
    print(f"总记录数: {total_records:,}")
    print(f"空文件数: {empty_count}")
    print(f"合并记录数: {merged_count:,}")
    print(f"验证结果: {verification}")
    print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print(f"输出文件: {full_vertex_lists}")
    print("="*60)
    
    # 可选：显示文件大小
    if os.path.exists(full_vertex_lists):
        file_size = os.path.getsize(full_vertex_lists) / (1024 * 1024 * 1024)  # GB
        print(f"输出文件大小: {file_size:.2f} GB")