"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

# get_concept_citation_ac_final.py
import glob
import gzip
import json
import os
import time
from datetime import datetime, date
import pickle
import random
import re
import sys
from typing import List, Tuple, Dict, Any, Set
from multiprocessing import Pool, cpu_count
import orjson
import ahocorasick  # 导入AC自动机库

# ==================== 文本处理函数 ====================

# 预编译替换规则
REPLACE_PAIRS = [
    ['\n', ' '], ['-', ' '], ['\"a', 'ae'], ['\"o', 'oe'], ['\"u', 'ue'],
    ['\' ', ''], ['\'', ''], ['  ', ' '], ['  ', ' ']
]

# 预编译正则表达式
CLEAN_PATTERNS = [(re.compile(re.escape(pair[0])), pair[1]) for pair in REPLACE_PAIRS]


def clean_text_fast(text: str) -> str:
    """快速文本清洗函数"""
    if not text:
        return ""
    for pattern, replacement in CLEAN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def get_single_article_string_optimized(article: Dict) -> str:
    """优化的文章文本提取函数"""
    # 获取标题
    curr_title = article.get('title', '')
    if curr_title is None:
        curr_title = ''
    
    # 处理摘要
    abstract_inverted_index = article.get('abstract_inverted_index', {})
    if abstract_inverted_index is None:
        abstract_inverted_index = {}
    
    if abstract_inverted_index:
        position_word_list = []
        for word, positions in abstract_inverted_index.items():
            if positions:
                for position in positions:
                    if position is not None:
                        position_word_list.append((position, word))
        
        if position_word_list:
            position_word_list.sort(key=lambda x: x[0])
            curr_abstract = ' '.join(word for _, word in position_word_list)
        else:
            curr_abstract = ''
    else:
        curr_abstract = article.get('abstract', '')
        if curr_abstract is None:
            curr_abstract = ''
    
    # 合并并清洗
    article_string = f"{str(curr_title)} {str(curr_abstract)}".lower().strip()
    if not article_string:
        return ""
    
    return clean_text_fast(article_string)


# ==================== 辅助函数 ====================

def extract_id_from_filename(filename: str, pattern: str) -> int:
    """从文件名中提取ID"""
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def get_date_and_part_from_path(path: str) -> Tuple[str, int]:
    """从路径中提取日期和部分号"""
    date_folder = os.path.dirname(path)
    date_str = date_folder.split('=')[-1]
    
    file_name = os.path.basename(path)
    part_str = file_name.split('_')[-1].split('.')[0]
    
    return date_str, int(part_str)


def build_ac_automaton(concepts: List[str]) -> ahocorasick.Automaton:
    """使用pyahocorasick构建AC自动机"""
    print(f"正在构建AC自动机，概念数: {len(concepts)}")
    start_time = time.time()
    
    # 创建自动机
    automaton = ahocorasick.Automaton()
    
    # 添加所有概念词
    for idx, concept in enumerate(concepts):
        concept_lower = concept.lower()
        automaton.add_word(concept_lower, idx)  # 存储概念ID
    
    # 构建失败指针
    automaton.make_automaton()
    
    build_time = time.time() - start_time
    print(f"AC自动机构建完成，耗时: {build_time:.2f}秒")
    
    # 不同版本的pyahocorasick有不同的获取大小的方法
    try:
        # 尝试使用size()方法
        node_count = automaton.size()
        print(f"总节点数: {node_count}")
    except AttributeError:
        try:
            # 尝试使用len()方法
            node_count = len(automaton)
            print(f"总节点数: {node_count}")
        except:
            # 如果都不行，就忽略
            pass
    
    return automaton


def load_concepts(concepts_file: str) -> Tuple[List[str], ahocorasick.Automaton]:
    """加载概念列表并构建AC自动机"""
    print(f"正在读取概念列表: {concepts_file}")
    with open(concepts_file, 'r', encoding='utf-8') as f:
        concepts = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"成功加载 {len(concepts)} 个概念")
    
    # 构建AC自动机
    automaton = build_ac_automaton(concepts)
    
    return concepts, automaton


def get_completed_files_from_output(output_folder: str, total_files: int) -> Set[int]:
    """从输出文件夹中获取已完成的文件ID（基于concept_part_XXX.gz文件）"""
    output_pattern = os.path.join(output_folder, 'concept_part_*.gz')
    
    completed_ids = set()
    for f in glob.glob(output_pattern):
        # 提取文件ID，例如 concept_part_042.gz -> 42
        match = re.search(r'concept_part_(\d+)\.gz', os.path.basename(f))
        if match:
            fid = int(match.group(1))
            if 0 <= fid < total_files:
                completed_ids.add(fid)
    
    return completed_ids


def process_single_file(args):
    """处理单个文件的函数（用于多进程）- 每个文件只处理前10000行"""
    file_path, file_id, automaton, paper_starting_date = args
    
    base_date = datetime(paper_starting_date.year, 
                        paper_starting_date.month, 
                        paper_starting_date.day).date()
    
    local_results = []
    processed = 0
    skipped = 0
    matched = 0
    file_start_time = time.time()
    
    # 设置每个文件最大处理行数
    MAX_LINES_PER_FILE = 1000
    
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                # 只处理前10000行数据
                if line_num >= MAX_LINES_PER_FILE:
                    print(f"  File {file_id}: 已达到最大处理行数 {MAX_LINES_PER_FILE}，停止处理")
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                try:
                    article = orjson.loads(line)
                    
                    # 检查必要字段
                    if 'publication_date' not in article:
                        skipped += 1
                        continue
                    
                    # 解析日期
                    try:
                        pub_date = datetime.strptime(
                            article['publication_date'], 
                            "%Y-%m-%d"
                        ).date()
                        days_since_start = (pub_date - base_date).days
                    except:
                        skipped += 1
                        continue
                    
                    # 获取引用信息
                    total_citations = article.get('cited_by_count', 0)
                    citations_by_year = article.get('counts_by_year', [])
                    
                    # 提取文章文本
                    article_text = get_single_article_string_optimized(article)
                    
                    if not article_text:
                        skipped += 1
                        continue
                    
                    # === AC自动机匹配（核心优化） ===
                    # 使用迭代器获取所有匹配
                    matched_concepts = set()
                    for end_idx, concept_id in automaton.iter(article_text):
                        matched_concepts.add(concept_id)
                    
                    # 记录结果
                    for concept_id in matched_concepts:
                        local_results.append([
                            concept_id,
                            days_since_start,
                            total_citations,
                            citations_by_year
                        ])
                        matched += 1
                    
                    processed += 1
                    
                    # 每处理1000行打印一次进度
                    if processed % 1000 == 0:
                        print(f"  File {file_id}: 已处理 {processed:,} 行, 匹配 {matched:,} 个")
                    
                except Exception as e:
                    skipped += 1
                    continue
        
        return {
            'file_id': file_id,
            'file_path': file_path,
            'edge_lists': local_results,
            'processed': processed,
            'skipped': skipped,
            'matched': matched,
            'time': time.time() - file_start_time,
            'total_lines': min(line_num + 1, MAX_LINES_PER_FILE) if 'line_num' in locals() else 0,
            'max_lines_limit': MAX_LINES_PER_FILE
        }
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return {
            'file_id': file_id,
            'file_path': file_path,
            'error': str(e)
        }


def save_results(results: List[Dict], 
                 vertex_folder: str,
                 vertex_folder_log: str,
                 log_folder: str):
    """保存处理结果，实现断点续写"""
    for result in results:
        if 'error' in result:
            file_id = result['file_id']
            error_log = os.path.join(log_folder, f'error_part_{file_id:03d}.txt')
            with open(error_log, 'w') as f:
                f.write(f"Error: {result['error']}\n")
                f.write(f"File: {result['file_path']}\n")
            print(f"文件 {file_id:03d} 处理出错: {result['error']}")
            continue
        
        file_id = result['file_id']
        formatted_id = f"{file_id:03d}"
        
        # 保存结果
        if result['edge_lists']:
            output_file = os.path.join(vertex_folder, f'concept_part_{formatted_id}.gz')
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(result['edge_lists'], f)
            
            # 保存详细日志
            log_file = os.path.join(vertex_folder_log, f'concept_part_{formatted_id}.txt')
            with open(log_file, 'w') as f:
                f.write(f"processed={result['processed']}\n")
                f.write(f"skipped={result['skipped']}\n")
                f.write(f"matched={result['matched']}\n")
                f.write(f"time={result['time']:.2f}s\n")
                f.write(f"concept_list={len(result['edge_lists'])}\n")
                f.write(f"total_lines={result.get('total_lines', 'N/A')}\n")
                f.write(f"max_lines_limit={result.get('max_lines_limit', 'N/A')}\n")
            
            print(f"文件 {formatted_id} 处理完成: "
                  f"{result['processed']:,} 行, {result['matched']:,} 匹配, "
                  f"耗时 {result['time']:.2f}s")
        else:
            # 空结果也保存一个空文件标记（避免重复处理）
            output_file = os.path.join(vertex_folder, f'concept_part_{formatted_id}.gz')
            # 保存空列表
            with gzip.open(output_file, 'wb') as f:
                pickle.dump([], f)
            
            # 记录空结果日志
            empty_log = os.path.join(vertex_folder_log, f'concept_part_{formatted_id}_empty.txt')
            with open(empty_log, 'w') as f:
                f.write(f"No matches found\n")
                f.write(f"processed={result['processed']}\n")
                f.write(f"skipped={result['skipped']}\n")
                f.write(f"total_lines={result.get('total_lines', 'N/A')}\n")
                f.write(f"max_lines_limit={result.get('max_lines_limit', 'N/A')}\n")
            
            print(f"文件 {formatted_id} 处理完成: 无匹配结果, "
                  f"{result['processed']:,} 行, 耗时 {result['time']:.2f}s")


def main():
    """主函数"""
    print("=" * 60)
    print("概念引用提取工具 (pyahocorasick极速版 - 每文件10000行限制)")
    print("断点续跑逻辑: 基于输出文件 concept_part_XXX.gz")
    print("=" * 60)
    
    # === 创建必要的文件夹 ===
    log_folder = 'logs_concept'
    vertex_folder = 'concept_citation'
    vertex_folder_log = 'concept_citation_log'
    
    for folder in [log_folder, vertex_folder, vertex_folder_log]:
        os.makedirs(folder, exist_ok=True)
        print(f"确保文件夹存在: {folder}")
    
    # === 加载概念列表并构建AC自动机 ===
    concepts_file = r"E:\study\research\ASIST\entities.txt"
    concepts, automaton = load_concepts(concepts_file)
    
    # === 设置论文数据路径 ===
    base_folder = "G:\\openalex-snapshot\\data\\works"
    date_pattern = 'updated_date=*'
    file_pattern = 'part_*.gz'
    
    print(f"\n搜索文件: {base_folder}/{date_pattern}/{file_pattern}")
    
    # 查找所有匹配的文件
    file_paths = glob.glob(f'{base_folder}/{date_pattern}/{file_pattern}')
    
    if not file_paths:
        print("错误: 未找到文件！")
        print(f"请检查路径: {base_folder}")
        sys.exit(1)
    
    print(f"找到 {len(file_paths)} 个文件")
    
    # 排序文件
    file_paths = sorted(file_paths, key=get_date_and_part_from_path)
    
    # 筛选时间范围
    start_date = datetime.strptime("2022-12-20", "%Y-%m-%d")
    end_date = datetime.strptime("2025-05-06", "%Y-%m-%d")
    
    curr_run_file_paths = []
    for path in file_paths:
        date_str, _ = get_date_and_part_from_path(path)
        file_date = datetime.strptime(date_str, "%Y-%m-%d")
        if start_date <= file_date <= end_date:
            curr_run_file_paths.append(path)
    
    print(f"筛选后 {len(curr_run_file_paths)} 个文件在时间范围内")
    
    if not curr_run_file_paths:
        print("错误: 没有符合条件的文件")
        sys.exit(1)
    
    # === 断点续写：检查输出文件夹中已存在的文件 ===
    completed_ids = get_completed_files_from_output(vertex_folder, len(curr_run_file_paths))
    
    print(f"\n断点续写检查 (基于输出文件 concept_part_XXX.gz):")
    print(f"  - 已存在输出文件: {len(completed_ids)} 个")
    print(f"  - 待处理文件: {len(curr_run_file_paths) - len(completed_ids)} 个")
    
    # 构建待处理文件列表
    files_to_process = []
    for i, path in enumerate(curr_run_file_paths):
        if i not in completed_ids:
            files_to_process.append((path, i))
        else:
            print(f"  跳过已完成文件: part_{i:03d}.gz (输出文件已存在)")
    
    if not files_to_process:
        print("\n所有文件都已处理完成！")
        with open("job_finish.txt", 'a') as f:
            f.write(f'\nFinish all: {datetime.now()}\n')
            f.write(f'Total files: {len(curr_run_file_paths)}\n')
            f.write(f'All files already processed\n')
        return
    
    print(f"\n开始处理 {len(files_to_process)} 个文件（每文件最多处理10000行）...")
    
    # === 参数设置 ===
    paper_starting_date = date(1990, 1, 1)
    
    # === 随机延时启动 ===
    delay = random.random() * 50
    print(f"延时启动 {delay:.1f} 秒...")
    time.sleep(delay)
    
    # === 使用多进程处理 ===
    num_processes = min(4, len(files_to_process)) 
    print(f"\n多进程配置:")
    print(f"  - CPU核心数: {cpu_count()}")
    print(f"  - 使用进程数: {num_processes}")
    print(f"  - 待处理文件: {len(files_to_process)}")
    print(f"  - 每文件最大行数: 10000")
    print(f"  - 断点续跑: 基于输出文件存在性检查")
    
    # 准备参数
    process_args = []
    for file_path, file_id in files_to_process:
        process_args.append((file_path, file_id, automaton, paper_starting_date))
    
    all_results = []
    start_total = time.time()
    
    # 使用进程池并行处理
    with Pool(processes=num_processes) as pool:
        # 使用imap_unordered实时获取结果
        for i, result in enumerate(pool.imap_unordered(process_single_file, process_args)):
            # 立即保存每个结果
            save_results([result], vertex_folder, vertex_folder_log, log_folder)
            all_results.append(result)
            
            # 打印进度
            progress = (i + 1) / len(process_args) * 100
            elapsed = time.time() - start_total
            eta = (elapsed / (i + 1)) * (len(process_args) - i - 1) if i > 0 else 0
            
            print(f"\n▶ 整体进度: {progress:.1f}% ({i+1}/{len(process_args)})")
            print(f"   已用时间: {elapsed/3600:.2f}小时, 预计剩余: {eta/3600:.2f}小时")
    
    total_time = time.time() - start_total
    
    # === 统计信息 ===
    total_processed = sum(r.get('processed', 0) for r in all_results)
    total_matched = sum(r.get('matched', 0) for r in all_results)
    total_skipped = sum(r.get('skipped', 0) for r in all_results)
    total_lines = sum(r.get('total_lines', 0) for r in all_results)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"总耗时: {total_time / 3600:.2f} 小时")
    print(f"总处理行数: {total_processed:,}")
    print(f"总匹配数: {total_matched:,}")
    print(f"总跳过数: {total_skipped:,}")
    print(f"文件总行数: {total_lines:,}")
    print(f"平均处理速度: {total_processed / total_time:.0f} 行/秒")
    print(f"每文件平均时间: {total_time / len(process_args):.1f} 秒")
    print(f"每文件最大行数限制: 10000")
    print(f"断点续跑: 基于输出文件")
    print("=" * 60)
    
    # 写入最终完成标记
    with open("job_finish.txt", 'a') as f:
        f.write(f'\nFinish all: {datetime.now()}\n')
        f.write(f'Total files in range: {len(curr_run_file_paths)}\n')
        f.write(f'Processed files in this run: {len(process_args)}\n')
        f.write(f'Total time: {total_time / 3600:.2f} hours\n')
        f.write(f'Processed lines: {total_processed:,}\n')
        f.write(f'Matched concepts: {total_matched:,}\n')
        f.write(f'Processing mode: Max 10000 lines per file\n')
        f.write(f'Resume logic: Based on output file existence\n')


if __name__ == "__main__":
    main()