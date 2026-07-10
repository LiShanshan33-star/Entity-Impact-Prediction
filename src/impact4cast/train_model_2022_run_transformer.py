"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

import os
import pickle
import gzip
import copy
import torch
from torch import nn
import torch.nn.functional as F
import random, time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Disable interactive windows, save all figures to files
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import rankdata
import networkx as nx
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, date
from itertools import combinations
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from general_utils import *
from preprocess_utils import *
from features_utils import *
from train_model_utils_transformer import *  # Use transformer version of utils
import pyarrow.dataset as ds
import gc
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ---- Random wait to avoid concurrent conflicts ----
rn_time = random.random() * 30
time.sleep(rn_time)


def load_large_parquet_in_chunks(file_path, batch_size=300_000, sample_frac=None, verbose=False):
    """
    Read large parquet file in batches, return pandas.DataFrame (concatenating all chunks).
    This is a convenience function; avoid concatenating all chunks into memory for very large files.
    """
    import pyarrow.parquet as pq

    if verbose:
        print(f"Loading {file_path} in chunks (batch_size={batch_size}) ...")
    pf = pq.ParquetFile(file_path)
    dfs = []
    total_rows = 0
    for i, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
        df_batch = batch.to_pandas()
        if sample_frac is not None and 0 < sample_frac < 1:
            df_batch = df_batch.sample(frac=sample_frac, random_state=42)
        # Compress data types to reduce memory usage (as needed)
        for col in df_batch.select_dtypes(include=["float64"]).columns:
            try:
                df_batch[col] = df_batch[col].astype("float32")
            except Exception:
                pass
        for col in df_batch.select_dtypes(include=["int64"]).columns:
            try:
                df_batch[col] = df_batch[col].astype("int32")
            except Exception:
                pass
        dfs.append(df_batch)
        total_rows += len(df_batch)
        if verbose:
            print(f"  loaded chunk {i+1}, rows so far: {total_rows}")
    if len(dfs) == 0:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    if verbose:
        print(f"Finished loading {file_path}, total rows: {len(df)}")
    return df


# ---------- memmap write utility (stream write large training table) ----------
def write_full_train_to_memmap(yes_df, no_parquet_path, memmap_path, batch_size=300_000, dtype=np.float32, verbose=True):
    """
    Write connected (yes_df) + unconnected (no_parquet_path) to disk memmap file in chunks.
    - If memmap_path exists and shape matches (rows, cols), return directly without rewriting (supports resume).
    - Returns (memmap_path, total_rows, ncols)
    """
    import pyarrow.parquet as pq

    # Ensure yes_df column order as baseline
    yes_cols = list(yes_df.columns)
    ncols = yes_df.shape[1]

    # Count unconnected total rows (from parquet metadata)
    pf = pq.ParquetFile(no_parquet_path)
    try:
        no_n = pf.metadata.num_rows
    except Exception:
        # Fall back to manual iteration
        no_n = 0
        for rg in range(pf.num_row_groups):
            no_n += pf.metadata.row_group(rg).num_rows

    yes_n = len(yes_df)
    total_n = int(yes_n + no_n)

    # If file exists and size matches, reuse directly
    if os.path.exists(memmap_path):
        try:
            mm_existing = np.memmap(memmap_path, dtype=dtype, mode='r')
            if mm_existing.size == total_n * ncols:
                if verbose:
                    print(f"✅ memmap already exists and matches shape: {memmap_path} rows={total_n} cols={ncols}. Reusing.")
                del mm_existing
                gc.collect()
                return memmap_path, total_n, ncols
            else:
                if verbose:
                    print(f"⚠️ memmap exists but size mismatch (will overwrite): {memmap_path}")
                del mm_existing
        except Exception:
            if verbose:
                print("⚠️ Check existing memmap encountered error, will rewrite.")

    # Create memmap file (may need disk space total_n * ncols * 4 bytes)
    if verbose:
        print(f"⏳ Creating memmap: {memmap_path}, rows={total_n}, cols={ncols}, estimated size={total_n * ncols * 4 / (1024**3):.2f} GiB")
    mm = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=(total_n, ncols))

    # Write yes part
    if yes_n > 0:
        mm[:yes_n, :] = yes_df.values.astype(dtype)
    offset = yes_n

    # chunkWrite no parquet
    written = 0
    for i, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
        df_chunk = batch.to_pandas()
        # Fill missing columns
        for col in yes_cols:
            if col not in df_chunk.columns:
                # Default values:num -> 0, citation_m -> 0.0, others -> 0
                if col == 'num':
                    df_chunk[col] = 0
                elif col == 'citation_m':
                    df_chunk[col] = 0.0
                else:
                    df_chunk[col] = 0
        df_chunk = df_chunk[yes_cols]  # Align column order
        arr = df_chunk.values.astype(dtype)
        mm[offset: offset + arr.shape[0], :] = arr
        offset += arr.shape[0]
        written += arr.shape[0]
        if verbose:
            print(f"  wrote chunk {i+1}, rows written so far: {written} (offset {offset})")
        # flush and free
        mm.flush()
        del df_chunk, arr
        gc.collect()

    # final flush
    mm.flush()
    del mm
    gc.collect()
    if verbose:
        print(f"✅ memmap written to {memmap_path}")
    return memmap_path, total_n, ncols


if __name__ == '__main__':
    split_type = 0  # 1 is for conditional case
    out_norm = False
    num_class = 2
    day_origin = date(1990, 1, 1)

    vertex_degree_cutoff = 1
    min_edges = 1
    years_delta = 3
    
    # Modified: using2019data for training, predicting2025year (2025 - 3 = 2022, but we need2019training data)
    # For predicting2025year, we use2019year as training baseyear (2019 + 3 = 2022, but actual prediction2025needs farther data)
    # Here we keep training logic unchanged but adjust target prediction year
    year_start = 2019  # Training base year set to2019
    
    graph_parameter = [year_start, years_delta, vertex_degree_cutoff, min_edges]

    # ---- Folder preparation ----
    save_folders, log_folder = make_folders(year_start, split_type, num_class, "train")
    # Ensure a folder for figures (all plots will be saved here)
    figs_folder = save_folders.get('figs') if isinstance(save_folders, dict) and 'figs' in save_folders else os.path.join(log_folder, "figures")
    os.makedirs(figs_folder, exist_ok=True)

    log_run = os.path.join(log_folder, f"train_model_{year_start + years_delta}_run_1_for_2025_prediction")

    with open(log_run + "_logs.txt", "a", encoding='utf-8') as myfile:
        myfile.write(f"\n\nstart: {datetime.now()} - Training with 2019 data for 2025 prediction\n")

    # Improvement: read parquet by columns and years parquet, reduce memory usage
    start_time = time.time()
    data_folder = "data_concept_graph"
    graph_file = os.path.join(data_folder, "full_dynamic_graph.parquet")

    # Time range to load, e.g. 2016-2019(includes years needed for training)
    years_to_load = list(range(2016, 2020))  # 2016-2019

    try:
        dataset = ds.dataset(graph_file, format="parquet")

        # Auto-detect which cYYYY columns
        available_cols = dataset.schema.names
        base_cols = ["v1", "v2", "time", "ct"]
        year_cols = [f"c{y}" for y in years_to_load if f"c{y}" in available_cols]
        cols_to_load = base_cols + year_cols

        print(f"Loadingcolumns: {cols_to_load}")

        table = dataset.to_table(columns=cols_to_load)
        full_dynamic_graph = table.to_pandas()

        del table, dataset
        gc.collect()

        print(f"✅ Data load complete, total: {len(full_dynamic_graph)} rows.")
    except Exception as e:
        print(f"⚠️ pyarrow.dataset load failed, falling back to batch read: {e}")
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(graph_file)
        dfs = []
        for batch in pf.iter_batches(batch_size=300_000):
            df_batch = batch.to_pandas()
            cols = [c for c in df_batch.columns if c in cols_to_load]
            df_batch = df_batch[cols]
            dfs.append(df_batch)
        full_dynamic_graph = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

    with open(log_run + "_logs.txt", "a", encoding='utf-8') as myfile:
        myfile.write(
            f"\n{datetime.now()}: Done, read full_dynamic_graph (cols {cols_to_load}): "
            f"{len(full_dynamic_graph)}; elapsed_time: {time.time() - start_time}\n"
        )

    # ---- Subsequent logic remains consistent ----
    feature_folder = "data_for_features"
    start_time = time.time()
    adj_mat_sparse = []
    node_neighbor_list = []
    num_neighbor_list = []

    for yy in [year_start, year_start - 1, year_start - 2]:
        data_file = os.path.join(feature_folder, f"adjacency_matrix_{yy}.gz")
        adj_mat = get_adjacency_matrix(full_dynamic_graph, yy, data_file)
        adj_mat_sparse.append(adj_mat)
        curr_node_neighbor = get_node_neighbor(adj_mat)
        node_neighbor_list.append(curr_node_neighbor)
        curr_num_neighbor = np.array(adj_mat.sum(axis=0)).flatten()
        num_neighbor_list.append(curr_num_neighbor)

    with open(log_run + "_logs.txt", "a", encoding='utf-8') as myfile:
        myfile.write(f"\n{datetime.now()}: Done, adjacency_matrix_sparse; elapsed_time: {time.time() - start_time}")

    start_time = time.time()
    vertex_features = get_all_node_feature(adj_mat_sparse, year_start, feature_folder)

    start_time = time.time()
    vc_feature_list = []
    for yy in [year_start, year_start - 1, year_start - 2]:
        data_file = os.path.join(feature_folder, f"concept_node_citation_data_{yy}.parquet")
        vc_df = pd.read_parquet(data_file)
        vc_feature = vc_df.values
        vc_feature_list.append(vc_feature)
        del vc_df
        gc.collect()

    vertex_cfeatures = get_all_node_cfeature(vc_feature_list)
    with open(log_run + "_logs.txt", "a", encoding='utf-8') as myfile:
        myfile.write(f"\n{datetime.now()}: Done, vertex_cfeatures; elapsed_time: {time.time() - start_time}")

    pair_cf_parameter = [vc_feature_list, node_neighbor_list, num_neighbor_list, vertex_features, vertex_cfeatures]

    # ---- Training set ----
    train_data_folder = 'data_pair_solution'
    
    # Note: these filenames may need adjustment based on actual data
    # Assuming we have2019training pair data
    train_pair_file1 = os.path.join(train_data_folder, f"unconnected_{year_start}_pair_solution_connected_{year_start + years_delta}.parquet")
    train_pair_file2 = os.path.join(train_data_folder, f"unconnected_{year_start}_pair_solution_unconnected_{year_start + years_delta}.parquet")

    # Use chunked loading for connected samples (usually smaller)
    time_start = time.time()
    train_pair_data_yes = load_large_parquet_in_chunks(train_pair_file1, batch_size=300_000, sample_frac=None, verbose=True)
    with open(log_run + "_logs.txt", "a", encoding='utf-8') as myfile:
        myfile.write(f"\nDone, read connected: {len(train_pair_data_yes)}; elapsed_time: {time.time() - time_start}")

    # Now use memmap streamWritecompleteTraining set(disk-backed), avoid OOM OOM
    memmap_file = os.path.join(train_data_folder, f"full_train_data_{year_start}_to_{year_start+years_delta}_for_2025.dat")
    memmap_meta_file = memmap_file + ".meta"
    try:
        memmap_path, total_rows, ncols = write_full_train_to_memmap(train_pair_data_yes, train_pair_file2, memmap_file, batch_size=300_000, dtype=np.float32, verbose=True)
    except Exception as e:
        print(f"❌ Write memmap failed: {e}")
        raise

    # as memmap format open full_train_data(will not load all into memory)
    full_train_data = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=(int(total_rows), int(ncols)))
    with open(log_run + "_logs.txt", "a", encoding='utf-8') as myfile:
        myfile.write(f"\n{datetime.now()}: Done, prepared full_train_data memmap: {memmap_path}; rows={total_rows}; cols={ncols}\n")

    # Release memory no longer needed
    try:
        del full_dynamic_graph
    except Exception:
        pass
    gc.collect()

    # ---- Validation set (we may not have 2022 validation data, but keep original validation logic) ----
    # Note: for2025year prediction,Validation setmay need adjustment
    eval_folder = "data_eval"
    start_time = time.time()
    
    # Try loading validation data, create empty array if not exists
    try:
        eval_file = os.path.join(eval_folder, "data_eval_pair_solution.parquet")
        if os.path.exists(eval_file):
            eval_data_features_df = pd.read_parquet(eval_file)
            eval_data_solution = eval_data_features_df.values
            del eval_data_features_df
            gc.collect()
            with open(log_run + "_logs.txt", "a", encoding='utf-8') as myfile:
                myfile.write(f"finish loading eval_data_features; {time.time() - start_time}")
        else:
            eval_data_solution = np.array([])
            print("WARNING: Validation set file not found, using empty array")
    except Exception as e:
        eval_data_solution = np.array([])
        print(f"WARNING: Failed to load validation set: {e}")

    start_time = time.time()
    try:
        eval_file = os.path.join(eval_folder, "eval_data_pair_feature.parquet")
        if os.path.exists(eval_file):
            eval_data_features_df = pd.read_parquet(eval_file)
            eval_data_features = eval_data_features_df.values
            del eval_data_features_df
            gc.collect()
            with open(log_run + "_logs.txt", "a", encoding='utf-8') as myfile:
                myfile.write(f"\nfinish loading eval_data_solution; {time.time() - start_time}")
        else:
            eval_data_features = np.array([])
            print("WARNING: Validation feature file not found, using empty array")
    except Exception as e:
        eval_data_features = np.array([])
        print(f"WARNING: Failed to load validation features: {e}")

    # ---- Training loop (does not change impact_classfication call interface) ----
    IR_start = 1
    IR_end = 40  # Can be adjusted as needed, original code uses 40
    IR_count = IR_start
    while IR_count <= IR_end:
        num_impact = random.randint(IR_start, IR_end)
        IR_num = [num_impact]
        IR_Str = format_IR(IR_num, split_type)

        logs_file_name = os.path.join(log_folder, f"train_model_{year_start + years_delta}_for_2025_" + IR_Str)
        if not os.path.exists(logs_file_name + "_logs.txt"):
            current_time = datetime.now()
            open(logs_file_name + "_logs.txt", 'a').close()

            batch_size = 1000
            lr_enc = 3 * 10 ** -5
            rnd_seed = 42
            hyper_parameter = [batch_size, lr_enc, rnd_seed]
            graph_parameter = [year_start, years_delta, vertex_degree_cutoff, min_edges]
            user_parameter = [num_class, IR_num, split_type, out_norm]

            # Pass memmap as full_train_data
            try:
                impact_classfication(
                    full_train_data,
                    eval_data_features,
                    eval_data_solution[:, 2] if len(eval_data_solution) > 0 else np.array([]),
                    pair_cf_parameter,
                    hyper_parameter,
                    graph_parameter,
                    user_parameter,
                    save_folders,
                    logs_file_name
                )
                print(f"DONE: Training complete for IR={num_impact}")
            except Exception as e:
                with open(logs_file_name + "_logs.txt", "a", encoding='utf-8') as myfile:
                    myfile.write(f"\nError when running impact_classfication for IR {IR_num}: {e}\n")
                print(f"ERROR: Error in impact_classfication for IR {IR_num}: {e}")
                IR_count += 1
                continue

            IR_count += 1
            rn_time = random.random() * 30
            time.sleep(rn_time)
        else:
            IR_count += 1
            continue

    with open(log_run + "_logs.txt", "a", encoding='utf-8') as myfile:
        myfile.write(f"\nfinish: {datetime.now()} - Completed training with 2019 data for 2025 prediction\n\n")
        
    print(f"🎉 All training complete! Models saved in {log_folder}")