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

#  ... 
log_folder = 'logs'
vertex_list_folder = 'concept_citation'
full_vertex_lists = os.path.join(vertex_list_folder, 'all_concept_citation.gz')
temp_folder = os.path.join(vertex_list_folder, 'temp_merge')

#  ... 
os.makedirs(log_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

log_files = 'log_merge_concept_citation.txt'

# UTF-8encoding ... 
log_file_path = os.path.join(log_folder, log_files)

#  ... 
with open(log_file_path, 'a', encoding='utf-8') as f:
    f.write(f'\n{"="*60}\n')
    f.write(f'Merge (2:  ... ): {datetime.now()}\n')
    f.write(f'{"="*60}\n')

#  ... Merge ... Merge ... 
list_file_names = os.listdir(vertex_list_folder)
vertex_file_name_unsorted = [file for file in list_file_names 
                            if file.endswith('.gz') and file != 'all_concept_citation.gz']
vertex_lists_files = sorted(vertex_file_name_unsorted)

print(f"Found {len(vertex_lists_files)} files to merge")
print(f" ... : {temp_folder}")
print(f" ... : 500,000  ... /")

# ==================== 2 ...  +  ...  ====================
def merge_files_with_temp():
    """ ... Merge ... """
    temp_files = []
    batch_size = 500000  # 50 ... 
    current_batch = []
    batch_index = 0
    total_records = 0
    empty_count = 0
    
    print("\n ... ...")
    print("-" * 50)
    
    for id_file, curr_vertex_file in enumerate(vertex_lists_files):
        file_path = os.path.join(vertex_list_folder, curr_vertex_file)
        
        try:
            #  ... 
            with gzip.open(file_path, 'rb') as f:
                vertex_data_list = pickle.load(f)
            
            file_record_count = len(vertex_data_list) if vertex_data_list else 0
            
            if vertex_data_list:
                #  ... 
                current_batch.extend(vertex_data_list)
                total_records += file_record_count
                
                #  ... 
                while len(current_batch) >= batch_size:
                    temp_file = os.path.join(temp_folder, f'temp_batch_{batch_index:04d}.pkl')
                    
                    #  ... batch_size ... 
                    batch_to_save = current_batch[:batch_size]
                    current_batch = current_batch[batch_size:]
                    
                    #  ... 
                    with open(temp_file, 'wb') as temp_f:
                        pickle.dump(batch_to_save, temp_f)
                    
                    temp_files.append(temp_file)
                    batch_index += 1
                    
                    print(f"  Create temp file: {os.path.basename(temp_file)} "
                          f"({len(batch_to_save):,}  ... )")
            else:
                empty_count += 1
                print(f'  Empty file: {curr_vertex_file}')
            
            # Progress
            progress = (id_file + 1) / len(vertex_lists_files) * 100
            print(f'File [{id_file+1}/{len(vertex_lists_files)}] {progress:.1f}%: '
                  f'{curr_vertex_file} ({file_record_count:,} ) -> '
                  f'Cumulative {total_records:,}  ... ')
            
            #  ... UTF-8encoding
            with open(log_file_path, 'a', encoding='utf-8') as log_f:
                log_f.write(f'File: {curr_vertex_file}; '
                           f'Record count: {file_record_count:,}; '
                           f'Cumulative: {total_records:,}; '
                           f'Progress: {progress:.1f}%; '
                           f'Temp files: {batch_index}\n')
                
        except Exception as e:
            error_msg = f"File {curr_vertex_file} error during: {e}"
            print(f"  ❌ {error_msg}")
            with open(log_file_path, 'a', encoding='utf-8') as log_f:
                log_f.write(f': {error_msg}\n')
    
    # Last batch
    if current_batch:
        temp_file = os.path.join(temp_folder, f'temp_batch_{batch_index:04d}_final.pkl')
        with open(temp_file, 'wb') as temp_f:
            pickle.dump(current_batch, temp_f)
        temp_files.append(temp_file)
        print(f"\n ... : {os.path.basename(temp_file)} "
              f"({len(current_batch):,}  ... )")
        batch_index += 1
    
    print(f"\n ...  {len(temp_files)}  ...  {total_records:,}  ... ")
    
    #  ... Merge ... 
    print("\n ... Merge temp files to final file...")
    print("-" * 50)
    
    merged_count = 0
    with gzip.open(full_vertex_lists, 'wb') as out_f:
        #  ... 
        for temp_file in sorted(temp_files):
            try:
                #  ... 
                with open(temp_file, 'rb') as in_f:
                    batch_data = pickle.load(in_f)
                
                #  ... 
                pickle.dump(batch_data, out_f)
                out_f.flush()
                
                merged_count += len(batch_data)
                print(f"  Merge: {os.path.basename(temp_file)} "
                      f"({len(batch_data):,} , Cumulative {merged_count:,} )")
                
                #  ... 
                os.remove(temp_file)
                
            except Exception as e:
                print(f"  ❌ Merge ...  {temp_file} error during: {e}")
    
    # Delete temp folder
    try:
        os.rmdir(temp_folder)
        print(f"\nDelete temp folder: {temp_folder}")
    except Exception as e:
        print(f"\nTemp folder not empty, keeping: {temp_folder}")
    
    return total_records, empty_count, merged_count


# ====================  ...  ====================
if __name__ == "__main__":
    print("="*60)
    print("Merge concept citation files (Plan 2: temp file batch processing)")
    print("="*60)
    
    #  ... Merge
    if not vertex_lists_files:
        print("No files found to merge!")
        sys.exit(1)
    
    
    
    # Merge
    start_time = time.time()
    total_records, empty_count, merged_count = merge_files_with_temp()
    elapsed_time = time.time() - start_time
    
    # Verification result ... 
    if total_records == merged_count:
        verification = "Passed (record count matches)"
    else:
        verification = f" ...  ( {abs(total_records - merged_count):,} )"
    
    #  ... UTF-8encoding
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f'\n{"="*60}\n')
        f.write(f'Merge: {datetime.now()}\n')
        f.write(f'Source file count: {len(vertex_lists_files)}\n')
        f.write(f'Temp files: {len(glob.glob(os.path.join(temp_folder, "temp_batch_*.pkl")))} \n')
        f.write(f'Record count: {total_records:,}\n')
        f.write(f'Empty file count: {empty_count}\n')
        f.write(f'Merged record count: {merged_count:,}\n')
        f.write(f'Verification result: {verification}\n')
        f.write(f' ... : {elapsed_time:.2f}  ({elapsed_time/60:.2f} )\n')
        f.write(f'Output file: {full_vertex_lists}\n')
        f.write(f'{"="*60}\n')
    
    print("\n" + "="*60)
    print("Merge ... ")
    print("="*60)
    print(f"Source file count: {len(vertex_lists_files)}")
    print(f"Record count: {total_records:,}")
    print(f"Empty file count: {empty_count}")
    print(f"Merged record count: {merged_count:,}")
    print(f"Verification result: {verification}")
    print(f" ... : {elapsed_time:.2f}  ({elapsed_time/60:.2f} )")
    print(f"Output file: {full_vertex_lists}")
    print("="*60)
    
    #  ... 
    if os.path.exists(full_vertex_lists):
        file_size = os.path.getsize(full_vertex_lists) / (1024 * 1024 * 1024)  # GB
        print(f"Output file: {file_size:.2f} GB")