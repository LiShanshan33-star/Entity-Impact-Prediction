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
import ahocorasick  # Import AC automaton library

# ==================== Text Processing Functions ====================

# Pre-compiled replacement rules
REPLACE_PAIRS = [
    ['\n', ' '], ['-', ' '], ['\"a', 'ae'], ['\"o', 'oe'], ['\"u', 'ue'],
    ['\' ', ''], ['\'', ''], ['  ', ' '], ['  ', ' ']
]

# Pre-compiled regex patterns
CLEAN_PATTERNS = [(re.compile(re.escape(pair[0])), pair[1]) for pair in REPLACE_PAIRS]


def clean_text_fast(text: str) -> str:
    """Fast text cleaning function."""
    if not text:
        return ""
    for pattern, replacement in CLEAN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def get_single_article_string_optimized(article: Dict) -> str:
    """Optimized article text extraction function."""
    # Get title
    curr_title = article.get('title', '')
    if curr_title is None:
        curr_title = ''
    
    # Process abstract
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
    
    # Combine and clean
    article_string = f"{str(curr_title)} {str(curr_abstract)}".lower().strip()
    if not article_string:
        return ""
    
    return clean_text_fast(article_string)


# ==================== Helper Functions ====================

def extract_id_from_filename(filename: str, pattern: str) -> int:
    """Extract ID from filename."""
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def get_date_and_part_from_path(path: str) -> Tuple[str, int]:
    """Extract date and part number from path."""
    date_folder = os.path.dirname(path)
    date_str = date_folder.split('=')[-1]
    
    file_name = os.path.basename(path)
    part_str = file_name.split('_')[-1].split('.')[0]
    
    return date_str, int(part_str)


def build_ac_automaton(concepts: List[str]) -> ahocorasick.Automaton:
    """Build AC automaton using pyahocorasick."""
    print(f"Building AC automaton, concept count: {len(concepts)}")
    start_time = time.time()
    
    # Create automaton
    automaton = ahocorasick.Automaton()
    
    # Add all concept terms
    for idx, concept in enumerate(concepts):
        concept_lower = concept.lower()
        automaton.add_word(concept_lower, idx)  # Store concept ID
    
    # Build failure pointers
    automaton.make_automaton()
    
    build_time = time.time() - start_time
    print(f"AC automaton build complete, elapsed: {build_time:.2f}s")
    
    # Different pyahocorasick versions have different size methods
    try:
        # Try size() method
        node_count = automaton.size()
        print(f"Total nodes: {node_count}")
    except AttributeError:
        try:
            # Try len() method
            node_count = len(automaton)
            print(f"Total nodes: {node_count}")
        except:
            # If neither works, ignore
            pass
    
    return automaton


def load_concepts(concepts_file: str) -> Tuple[List[str], ahocorasick.Automaton]:
    """Load concept list and build AC automaton."""
    print(f"Reading concept list: {concepts_file}")
    with open(concepts_file, 'r', encoding='utf-8') as f:
        concepts = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Successfully loaded {len(concepts)} concepts")
    
    # Build AC automaton
    automaton = build_ac_automaton(concepts)
    
    return concepts, automaton


def get_completed_files_from_output(output_folder: str, total_files: int) -> Set[int]:
    """Get completed file IDs from output folder (based on concept_part_XXX.gz files)."""
    output_pattern = os.path.join(output_folder, 'concept_part_*.gz')
    
    completed_ids = set()
    for f in glob.glob(output_pattern):
        # Extract file ID, e.g. concept_part_042.gz -> 42
        match = re.search(r'concept_part_(\d+)\.gz', os.path.basename(f))
        if match:
            fid = int(match.group(1))
            if 0 <= fid < total_files:
                completed_ids.add(fid)
    
    return completed_ids


def process_single_file(args):
    """Process a single file (for multiprocessing) - max 10000 lines per file."""
    file_path, file_id, automaton, paper_starting_date = args
    
    base_date = datetime(paper_starting_date.year, 
                        paper_starting_date.month, 
                        paper_starting_date.day).date()
    
    local_results = []
    processed = 0
    skipped = 0
    matched = 0
    file_start_time = time.time()
    
    # Set max lines per file
    MAX_LINES_PER_FILE = 1000
    
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                # Only process first 10000 lines
                if line_num >= MAX_LINES_PER_FILE:
                    print(f"  File {file_id}: Reached max lines {MAX_LINES_PER_FILE}, stopping")
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                try:
                    article = orjson.loads(line)
                    
                    # Check required fields
                    if 'publication_date' not in article:
                        skipped += 1
                        continue
                    
                    # Parse date
                    try:
                        pub_date = datetime.strptime(
                            article['publication_date'], 
                            "%Y-%m-%d"
                        ).date()
                        days_since_start = (pub_date - base_date).days
                    except:
                        skipped += 1
                        continue
                    
                    # Get citation info
                    total_citations = article.get('cited_by_count', 0)
                    citations_by_year = article.get('counts_by_year', [])
                    
                    # Extract article text
                    article_text = get_single_article_string_optimized(article)
                    
                    if not article_text:
                        skipped += 1
                        continue
                    
                    # === AC automaton matching (core optimization) ===
                    # Use iterator to get all matches
                    matched_concepts = set()
                    for end_idx, concept_id in automaton.iter(article_text):
                        matched_concepts.add(concept_id)
                    
                    # Record results
                    for concept_id in matched_concepts:
                        local_results.append([
                            concept_id,
                            days_since_start,
                            total_citations,
                            citations_by_year
                        ])
                        matched += 1
                    
                    processed += 1
                    
                    # Print progress every 1000 lines
                    if processed % 1000 == 0:
                        print(f"  File {file_id}: Processed {processed:,} lines, matched {matched:,}")
                    
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
        print(f"Error processing file {file_path}: {e}")
        return {
            'file_id': file_id,
            'file_path': file_path,
            'error': str(e)
        }


def save_results(results: List[Dict], 
                 vertex_folder: str,
                 vertex_folder_log: str,
                 log_folder: str):
    """Save processing results, support resume."""
    for result in results:
        if 'error' in result:
            file_id = result['file_id']
            error_log = os.path.join(log_folder, f'error_part_{file_id:03d}.txt')
            with open(error_log, 'w') as f:
                f.write(f"Error: {result['error']}\n")
                f.write(f"File: {result['file_path']}\n")
            print(f"File {file_id:03d} error: {result['error']}")
            continue
        
        file_id = result['file_id']
        formatted_id = f"{file_id:03d}"
        
        # Save results
        if result['edge_lists']:
            output_file = os.path.join(vertex_folder, f'concept_part_{formatted_id}.gz')
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(result['edge_lists'], f)
            
            # Save detailed log
            log_file = os.path.join(vertex_folder_log, f'concept_part_{formatted_id}.txt')
            with open(log_file, 'w') as f:
                f.write(f"processed={result['processed']}\n")
                f.write(f"skipped={result['skipped']}\n")
                f.write(f"matched={result['matched']}\n")
                f.write(f"time={result['time']:.2f}s\n")
                f.write(f"concept_list={len(result['edge_lists'])}\n")
                f.write(f"total_lines={result.get('total_lines', 'N/A')}\n")
                f.write(f"max_lines_limit={result.get('max_lines_limit', 'N/A')}\n")
            
            print(f"File {formatted_id} done: "
                  f"{result['processed']:,} lines, {result['matched']:,} matches, "
                  f"elapsed {result['time']:.2f}s")
        else:
            # Save empty file marker for empty results (avoid reprocessing)
            output_file = os.path.join(vertex_folder, f'concept_part_{formatted_id}.gz')
            # Save empty list
            with gzip.open(output_file, 'wb') as f:
                pickle.dump([], f)
            
            # Log empty results
            empty_log = os.path.join(vertex_folder_log, f'concept_part_{formatted_id}_empty.txt')
            with open(empty_log, 'w') as f:
                f.write(f"No matches found\n")
                f.write(f"processed={result['processed']}\n")
                f.write(f"skipped={result['skipped']}\n")
                f.write(f"total_lines={result.get('total_lines', 'N/A')}\n")
                f.write(f"max_lines_limit={result.get('max_lines_limit', 'N/A')}\n")
            
            print(f"File {formatted_id} done: no matches, "
                  f"{result['processed']:,} lines, elapsed {result['time']:.2f}s")


def main():
    """Main function."""
    print("=" * 60)
    print("Concept Citation Extraction Tool (pyahocorasick fast edition - 10000 lines per file limit)")
    print("Resume logic: based on output file concept_part_XXX.gz")
    print("=" * 60)
    
    # === Create necessary folders ===
    log_folder = 'logs_concept'
    vertex_folder = 'concept_citation'
    vertex_folder_log = 'concept_citation_log'
    
    for folder in [log_folder, vertex_folder, vertex_folder_log]:
        os.makedirs(folder, exist_ok=True)
        print(f"Ensuring folder exists: {folder}")
    
    # === Load concept list and build AC automaton ===
    concepts_file = r"./data/entities.txt"
    concepts, automaton = load_concepts(concepts_file)
    
    # === Set paper data paths ===
    base_folder = "./data/openalex/works"
    date_pattern = 'updated_date=*'
    file_pattern = 'part_*.gz'
    
    print(f"\nSearching files: {base_folder}/{date_pattern}/{file_pattern}")
    
    # Find all matching files
    file_paths = glob.glob(f'{base_folder}/{date_pattern}/{file_pattern}')
    
    if not file_paths:
        print("Error: No files found!")
        print(f"Check path: {base_folder}")
        sys.exit(1)
    
    print(f"Found {len(file_paths)} files")
    
    # Sort files
    file_paths = sorted(file_paths, key=get_date_and_part_from_path)
    
    # Filter by time range
    start_date = datetime.strptime("2022-12-20", "%Y-%m-%d")
    end_date = datetime.strptime("2025-05-06", "%Y-%m-%d")
    
    curr_run_file_paths = []
    for path in file_paths:
        date_str, _ = get_date_and_part_from_path(path)
        file_date = datetime.strptime(date_str, "%Y-%m-%d")
        if start_date <= file_date <= end_date:
            curr_run_file_paths.append(path)
    
    print(f"After filtering: {len(curr_run_file_paths)} files in time range")
    
    if not curr_run_file_paths:
        print("Error: No matching files")
        sys.exit(1)
    
    # === Resume: check existing files in output folder ===
    completed_ids = get_completed_files_from_output(vertex_folder, len(curr_run_file_paths))
    
    print(f"\nResume check (based on output files concept_part_XXX.gz):")
    print(f"  - Existing output files: {len(completed_ids)}")
    print(f"  - Files to process: {len(curr_run_file_paths) - len(completed_ids)}")
    
    # Build list of files to process
    files_to_process = []
    for i, path in enumerate(curr_run_file_paths):
        if i not in completed_ids:
            files_to_process.append((path, i))
        else:
            print(f"  Skipping completed file: part_{i:03d}.gz (output exists)")
    
    if not files_to_process:
        print("\nAll files already processed!")
        with open("job_finish.txt", 'a') as f:
            f.write(f'\nFinish all: {datetime.now()}\n')
            f.write(f'Total files: {len(curr_run_file_paths)}\n')
            f.write(f'All files already processed\n')
        return
    
    print(f"\nProcessing {len(files_to_process)} files (max 10000 lines per file)...")
    
    # === Parameter setup ===
    paper_starting_date = date(1990, 1, 1)
    
    # === Random delay start ===
    delay = random.random() * 50
    print(f"Delaying start {delay:.1f}s...")
    time.sleep(delay)
    
    # === Multiprocessing ===
    num_processes = min(4, len(files_to_process)) 
    print(f"\nMultiprocess config:")
    print(f"  - CPU cores: {cpu_count()}")
    print(f"  - Processes used: {num_processes}")
    print(f"  - Files to process: {len(files_to_process)}")
    print(f"  - Max lines per file: 10000")
    print(f"  - Resume: based on output file existence check")
    
    # Prepare args
    process_args = []
    for file_path, file_id in files_to_process:
        process_args.append((file_path, file_id, automaton, paper_starting_date))
    
    all_results = []
    start_total = time.time()
    
    # Use process pool for parallel processing
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for real-time results
        for i, result in enumerate(pool.imap_unordered(process_single_file, process_args)):
            # Save each result immediately
            save_results([result], vertex_folder, vertex_folder_log, log_folder)
            all_results.append(result)
            
            # Print progress
            progress = (i + 1) / len(process_args) * 100
            elapsed = time.time() - start_total
            eta = (elapsed / (i + 1)) * (len(process_args) - i - 1) if i > 0 else 0
            
            print(f"\n> Overall progress: {progress:.1f}% ({i+1}/{len(process_args)})")
            print(f"   Elapsed: {elapsed/3600:.2f}h, ETA: {eta/3600:.2f}h")
    
    total_time = time.time() - start_total
    
    # === Summary statistics ===
    total_processed = sum(r.get('processed', 0) for r in all_results)
    total_matched = sum(r.get('matched', 0) for r in all_results)
    total_skipped = sum(r.get('skipped', 0) for r in all_results)
    total_lines = sum(r.get('total_lines', 0) for r in all_results)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Total lines processed: {total_processed:,}")
    print(f"Total matches: {total_matched:,}")
    print(f"Total skipped: {total_skipped:,}")
    print(f"Total file lines: {total_lines:,}")
    print(f"Average processing speed: {total_processed / total_time:.0f} lines/sec")
    print(f"Average time per file: {total_time / len(process_args):.1f}s")
    print(f"Max lines per file limit: 10000")
    print(f"Resume: based on output file")
    print("=" * 60)
    
    # Write final completion marker
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