"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

import os
import pickle
import gzip
from datetime import datetime, date
import torch
from torch import nn
import torch.nn.functional as F
import random, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from collections import defaultdict, Counter
from itertools import combinations
import pandas as pd
import copy
from features_utils import *
from preprocess_utils import *
from features_utils import *
from general_utils import *

#####--------------Transformer 架构----------------------------#####

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class FeatureProjection(nn.Module):
    """特征投影层：将141维特征投影到Transformer维度"""
    def __init__(self, input_dim, d_model):
        super(FeatureProjection, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.projection(x)


class FeatureAggregator(nn.Module):
    """特征聚合器：将多个特征进行聚合"""
    def __init__(self, d_model, num_heads=4):
        super(FeatureAggregator, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key_value):
        # query: [seq_len, batch, d_model]
        # key_value: [seq_len, batch, d_model]
        attn_output, _ = self.cross_attn(query, key_value, key_value)
        return self.norm(query + attn_output)


class TransformerNetwork(nn.Module):
    """
    Transformer架构的神经网络 - 修复维度问题
    """
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=1024, dropout=0.1, output_size=1):
        super(TransformerNetwork, self).__init__()
        
        self.d_model = d_model
        self.input_size = input_size
        self.output_size = output_size
        
        # 特征投影
        self.feature_proj = FeatureProjection(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            )
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        # 特征聚合层
        self.feature_aggregator = FeatureAggregator(d_model)
        
        # 输出层 - 修复：确保输出维度正确
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_size)
        )
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 将特征投影到Transformer维度 [batch, features] -> [batch, d_model]
        x = self.feature_proj(x)  # [batch, d_model]
        
        # 重塑为序列格式 [seq_len=1, batch, d_model]
        x = x.unsqueeze(0)  # [1, batch, d_model]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 通过Transformer编码器层
        for layer in self.transformer_encoder:
            x = layer(x)
        
        # 特征聚合
        x = self.feature_aggregator(x, x)
        
        # 移除序列维度 [1, batch, d_model] -> [batch, d_model]
        x = x.squeeze(0)  # [batch, d_model]
        
        # 输出层 - 修复：确保输出形状正确
        output = self.output_layer(x)  # [batch, output_size]
        
        # 如果是二分类且output_size=1，压缩最后一维
        if self.output_size == 1:
            output = output.squeeze(1)  # [batch]
        
        return output


def train_model(model_semnet, device, train_input_data, test_input_data, 
                hyper_parameter, graph_parameter, user_parameter, 
                save_net_folder, logs_file_name):
    """
    训练Transformer模型 - 修复维度问题
    """
    year_start, years_delta, vertex_degree_cutoff, min_edges = graph_parameter
    batch_size, lr_enc, rnd_seed = hyper_parameter
    num_class, IR_num, split_type, out_norm = user_parameter

    IR_Str = format_IR(IR_num, split_type)
    size_of_loss_check = min(1000, len(train_input_data[0]) if len(train_input_data) > 0 else 1000)
    
    optimizer_predictor = torch.optim.AdamW(model_semnet.parameters(), lr=lr_enc, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_predictor, mode='min', factor=0.5, patience=10
    )
    
    train_data = []
    test_data = []
    for ii in range(len(train_input_data)):
        train_data_tensor = torch.tensor(train_input_data[ii], dtype=torch.float).to(device)
        train_data.append(train_data_tensor)
        test_data_tensor = torch.tensor(test_input_data[ii], dtype=torch.float).to(device)
        test_data.append(test_data_tensor)
    
    test_loss_total = []
    train_loss_total = []
    moving_avg = []
    criterion = torch.nn.MSELoss()
    start_time = time.time()
    
    for iteration in range(50000):
        model_semnet.train()
        data_sets = train_data
        total_loss = 0
        
        for idx_dataset in range(len(data_sets)):
            # 确保batch_size不超过数据集大小
            current_batch_size = min(batch_size, len(data_sets[idx_dataset]))
            idx = torch.randint(0, len(data_sets[idx_dataset]), (current_batch_size,))
            data_train_samples = data_sets[idx_dataset][idx]
            
            # 前向传播
            calc_properties = model_semnet(data_train_samples)  # 已经是squeeze后的结果
            
            # 修复：确保目标标签维度匹配
            if num_class <= 2:
                curr_pred_one_hot = torch.tensor([idx_dataset] * current_batch_size, dtype=torch.float).to(device)
            else:
                curr_pred = torch.tensor([idx_dataset] * current_batch_size, dtype=torch.long).to(device)
                curr_pred_one_hot = F.one_hot(curr_pred, num_classes=num_class).float().to(device)
            
            real_loss = criterion(calc_properties, curr_pred_one_hot)
            total_loss += torch.clamp(real_loss, min=0., max=50000.).double()

        optimizer_predictor.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model_semnet.parameters(), max_norm=1.0)
        
        optimizer_predictor.step()
        
        # 评估
        with torch.no_grad():
            model_semnet.eval()
            eval_datasets = flatten([train_data, test_data])
            all_real_loss = []
            
            for idx_dataset in range(len(eval_datasets)):
                # 确保评估时不会超出范围
                eval_size = min(size_of_loss_check, len(eval_datasets[idx_dataset]))
                if eval_size == 0:
                    continue
                    
                calc_properties = model_semnet(eval_datasets[idx_dataset][0:eval_size])
                
                if num_class <= 2:
                    curr_pred_one_hot = torch.tensor(
                        [idx_dataset % num_class] * eval_size, 
                        dtype=torch.float
                    ).to(device)
                else:
                    curr_pred = torch.tensor(
                        [idx_dataset % num_class] * eval_size, 
                        dtype=torch.long
                    ).to(device)
                    curr_pred_one_hot = F.one_hot(curr_pred, num_classes=num_class).float().to(device)
                
                real_loss = criterion(calc_properties, curr_pred_one_hot)
                all_real_loss.append(real_loss.detach().cpu().numpy())
            
            if len(all_real_loss) >= 2*num_class:
                train_loss_total.append(sum(all_real_loss[:num_class]))
                test_loss_total.append(sum(all_real_loss[num_class:2 * num_class]))
                
                # 调整学习率
                scheduler.step(test_loss_total[-1])

                if iteration % 500 == 0:
                    train_loss_number = sum(all_real_loss[:num_class])
                    test_loss_number = sum(all_real_loss[num_class:2 * num_class])
                    
                    current_lr = optimizer_predictor.param_groups[0]['lr']
                    print(f'    train_model (Transformer): iteration: {iteration} - train loss: {train_loss_number:.4f}; '
                          f'test loss: {test_loss_number:.4f}; lr: {current_lr:.2e}; time: {time.time()-start_time:.2f}s')
                    
                    with open(logs_file_name + "_logs.txt", "a") as myfile:
                        myfile.write(f'\n    train_model (Transformer): iteration: {iteration} - '
                                    f'train loss: {train_loss_number:.4f}; test loss: {test_loss_number:.4f}; '
                                    f'time: {time.time()-start_time:.2f}s')
                    start_time = time.time()

                # 保存最佳模型
                if len(test_loss_total) > 0 and test_loss_total[-1] == min(test_loss_total):
                    model_semnet.eval()
                    net_file = os.path.join(save_net_folder, f"transformer_net_year_{year_start}_delta_{years_delta}_class_{num_class}_{IR_Str}.pt")
                    net_state_file = os.path.join(save_net_folder, f"transformer_state_year_{year_start}_delta_{years_delta}_class_{num_class}_{IR_Str}.pt")
                    torch.save(model_semnet, net_file)
                    torch.save(model_semnet.state_dict(), net_state_file)
                    model_semnet.train()

                # 早停
                if len(test_loss_total) > 1000:
                    test_loss_moving_avg = sum(test_loss_total[-500:])
                    moving_avg.append(test_loss_moving_avg)
                    if len(moving_avg) > 1000:
                        if (moving_avg[-1] > moving_avg[-25] and 
                            moving_avg[-1] > moving_avg[-175] and 
                            moving_avg[-1] > moving_avg[-350] and 
                            moving_avg[-1] > moving_avg[-750] and 
                            moving_avg[-1] > moving_avg[-950]):
                            print('    Early stopping kicked in')
                            break
    
    return train_loss_total, test_loss_total, moving_avg


def plot_train_loss(train_loss_total, test_loss_total, moving_avg, 
                    graph_parameter, user_parameter, store_file):
    """
    绘制训练损失曲线
    """
    year_start, years_delta, vertex_degree_cutoff, min_edges = graph_parameter
    num_class, IR_num, split_type, out_norm = user_parameter
    
    plt.figure(figsize=(10, 6))
    if len(train_loss_total) > 0:
        plt.plot(train_loss_total, label='Train Loss', alpha=0.7)
    if len(test_loss_total) > 0:
        plt.plot(test_loss_total, label='Test Loss', alpha=0.7)
    plt.title(f"Transformer Loss (start={year_start}, class={num_class}, IR={IR_num})")
    plt.xlabel('Iterations (×100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(store_file, dpi=300, bbox_inches='tight')
    plt.close()


def eval_model_in_batches(model, device, data_batch, data_feature, user_parameter):
    """
    批量评估Transformer模型
    """
    num_class, IR_num, split_type, out_norm = user_parameter
    
    if not torch.is_tensor(data_feature):
        tensor_feature = torch.tensor(data_feature, dtype=torch.float).to(device)
    
    output_batches = []
    
    for start_i in range(0, len(tensor_feature), data_batch):
        end_i = min(start_i + data_batch, len(tensor_feature))
        batch_X = tensor_feature[start_i:end_i]
        batch_output = model(batch_X).detach()
        output_batches.append(batch_output)
    
    outputs = torch.cat(output_batches)
    outputs = outputs.cpu().numpy()
    
    # 确保输出是一维数组
    if len(outputs.shape) > 1 and outputs.shape[1] == 1:
        outputs = outputs.flatten()
    
    if out_norm and num_class > 2:
        nn_outputs = F.softmax(torch.tensor(outputs), dim=1).numpy()
    else:
        nn_outputs = outputs
    
    return nn_outputs


# 主函数保持不变
def impact_classfication(full_train_data, data_feature_eval, solution_eval, 
                         pair_cf_parameter, hyper_parameter, graph_parameter, 
                         user_parameter, save_folders, logs_file_name):
    """
    使用Transformer进行影响因子分类
    """
    node_cfeature_list, node_neighbor_list, num_neighbor_list, node_feature, node_cfeature = pair_cf_parameter
    batch_size, lr_enc, rnd_seed = hyper_parameter
    year_start, years_delta, vertex_degree_cutoff, min_edges = graph_parameter
    num_class, IR_num, split_type, out_norm = user_parameter

    save_net_folder, save_loss_folder, save_figure_folder, save_result_folder = save_folders
    IR_Str = format_IR(IR_num, split_type)
    
    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)

    # 准备训练和测试数据
    with open(logs_file_name + "_logs.txt", "a") as myfile:
        myfile.write(f"\n1.1) {datetime.now()}: Prepare train and test data (Transformer)...")

    data_subset = prepare_split_datasets(full_train_data, user_parameter, logs_file_name)
    train_valid_test_size = [0.85, 0.15, 0.0]
    dataset_train, dataset_test = shuffle_split_datasets(data_subset, train_valid_test_size)
    
    pair_train, solution_train = get_pair_solution_datasets(dataset_train, hyper_parameter, user_parameter, logs_file_name)
    pair_test, solution_test = get_pair_solution_datasets(dataset_test, hyper_parameter, user_parameter, logs_file_name)
    
    # 训练特征
    pair_feature_train, pair_cfeature_train = get_all_pair_features(
        node_cfeature_list, node_neighbor_list, num_neighbor_list, pair_train, logs_file_name
    )
    node_pair_feature_train = [node_feature, node_cfeature, pair_feature_train, pair_cfeature_train]
    data_feature_train, train_input_data = prepare_train_data(
        pair_train, solution_train, node_pair_feature_train, user_parameter, logs_file_name
    )
    
    # 测试特征
    pair_feature_test, pair_cfeature_test = get_all_pair_features(
        node_cfeature_list, node_neighbor_list, num_neighbor_list, pair_test, logs_file_name
    )
    node_pair_feature_test = [node_feature, node_cfeature, pair_feature_test, pair_cfeature_test]
    data_feature_test, test_input_data = prepare_train_data(
        pair_test, solution_test, node_pair_feature_test, user_parameter, logs_file_name
    )

    # 训练Transformer
    print(f"\n1.2) {datetime.now()}: Train Transformer Neural Network...")
    with open(logs_file_name + "_logs.txt", "a") as myfile:
        myfile.write(f"\n1.2) {datetime.now()}: Train Transformer Neural Network...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = len(data_feature_train[0])  # 141
    
    # Transformer参数
    d_model = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 1024
    dropout = 0.1
    
    if num_class <= 2:
        output_size = 1
    else:
        output_size = num_class
    
    # 使用修复后的Transformer模型
    model_semnet = TransformerNetwork(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_size=output_size
    ).to(device)
    
    print(f"    Transformer parameters: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    print(f"    Total parameters: {sum(p.numel() for p in model_semnet.parameters()):,}")
    
    model_semnet.train()
    train_loss, test_loss, moving_avg = train_model(
        model_semnet, device, train_input_data, test_input_data, 
        hyper_parameter, graph_parameter, user_parameter, 
        save_net_folder, logs_file_name
    )
    
    store_name = os.path.join(
        save_loss_folder, 
        f"transformer_loss_plot_year_{year_start}_delta_{years_delta}_class_{num_class}_{IR_Str}.png"
    )
    plot_train_loss(train_loss, test_loss, moving_avg, graph_parameter, user_parameter, store_name)

    # 计算AUC
    print(f'\n1.3) {datetime.now()}: Computes the AUC for training and test data (Transformer)...')
    with open(logs_file_name + "_logs.txt", "a") as myfile:
        myfile.write(f"\n1.3) {datetime.now()}: Computes the AUC for training and test data (Transformer)...")

    model_semnet.eval()
    data_batch_size = batch_size
    
    roc_curve_name = [
        f"Transformer_Train_Year_{year_start}_Delta_{years_delta}_Class_{num_class}_ROC_{IR_Str}.png",
        f"Transformer_Test_Year_{year_start}_Delta_{years_delta}_Class_{num_class}_ROC_{IR_Str}.png",
        f"Transformer_Eval_Year_{year_start}_Delta_{years_delta}_Class_{num_class}_ROC_{IR_Str}.png"
    ]
    
    # 训练集AUC
    output_nn = eval_model_in_batches(model_semnet, device, data_batch_size, data_feature_train, user_parameter)
    solution_arr = classify_solution(solution_train, user_parameter)
    train_auc_score = calculate_plot_ROC(solution_arr, output_nn, user_parameter, roc_curve_name[0], save_figure_folder)
    
    # 测试集AUC
    output_nn = eval_model_in_batches(model_semnet, device, data_batch_size, data_feature_test, user_parameter)
    solution_arr = classify_solution(solution_test, user_parameter)
    test_auc_score = calculate_plot_ROC(solution_arr, output_nn, user_parameter, roc_curve_name[1], save_figure_folder)

    # 验证集AUC
    if len(data_feature_eval) > 0:
        print(f'\n1.4) {datetime.now()}: Evaluate the AUC for future data (Transformer)...')
        with open(logs_file_name + "_logs.txt", "a") as myfile:
            myfile.write(f"\n1.4) {datetime.now()}: Evaluate the AUC for future data (Transformer)...")

        output_nn = eval_model_in_batches(model_semnet, device, data_batch_size, data_feature_eval, user_parameter)
        solution_arr = classify_solution(solution_eval, user_parameter)
        eval_auc_score = calculate_plot_ROC(solution_arr, output_nn, user_parameter, roc_curve_name[2], save_figure_folder)
    else:
        eval_auc_score = []

    # 保存结果
    store_folder = os.path.join(
        save_result_folder, 
        f'Transformer_AUC_Report_Year_{year_start}_Class_{num_class}_{IR_Str}.txt'
    )
    with open(store_folder, 'a') as f:
        f.write(f"IR={IR_num}: train={train_auc_score}; test={test_auc_score}; eval={eval_auc_score}\n")
    
    with open(os.path.join(save_result_folder, f"Transformer_All_AUC_Report_Year_Train_{year_start}.txt"), 'a') as f:
        f.write(f"IR={IR_num}: train={train_auc_score}; test={test_auc_score}; eval={eval_auc_score}\n")
    
    print(f"Transformer - IR={IR_num}: train={train_auc_score}; test={test_auc_score}; eval={eval_auc_score}")
    
    return True