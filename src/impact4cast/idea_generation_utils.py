"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

import ollama
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import sys

# ========== 统一调用函数 (核心修改) ==========
def ask_deepseek_conversational(messages, model="deepseek-r1:7b"):
    """
    统一封装 Ollama API 调用 (已修改)。
    
    此函数现在接收一个 *完整的消息历史* (messages 列表)，
    并将其发送给 Ollama。
    它返回助手的新响应内容。
    
    Args:
        messages (list): 遵循 Ollama/OpenAI 格式的消息历史列表。
        model (str): 要使用的模型名称。
        
    Returns:
        str: 助手返回的最新消息内容。
    """
    try:
        response = ollama.chat(
            model=model,
            messages=messages  # 传入完整的对话历史
        )
        # 返回助手的 *新* 响应内容
        return response["message"]["content"]
    except Exception as e:
        return f"[Error: {e}]"

# ========== 多阶段*对话*核心逻辑 (核心修改) ==========
def generate_project_idea_stateful(concept1, concept2, model="deepseek-r1:7b"):
    """
    执行四阶段 LLM *对话* (有记忆)，为单个概念对生成详细的项目构想。
    这会维持一个统一的对话历史。
    
    Args:
        concept1 (str): 第一个概念。
        concept2 (str): 第二个概念。
        model (str): 使用的模型名称。
        
    Returns:
        str: 包含所有阶段结果的格式化字符串。
    """
    
    # 1. 对话历史 (关键)
    # 我们将在这里累积所有的对话
    messages = []
    
    # 2. 最终的文本文档
    # (用于保存和打印，格式与你原来的一致)
    all_text = []

    # --- 内部辅助函数，用于运行一个对话步骤 ---
    def run_step(prompt_text, header_text):
        """
        一个辅助函数，用于：
        1. 将用户的 prompt 添加到历史
        2. 调用 API 获取响应
        3. 将助手的响应添加到历史
        4. 将助手的响应计入最终的文本文档
        """
        nonlocal messages, all_text
        
        # 1. 将用户的新 prompt 添加到历史
        messages.append({"role": "user", "content": prompt_text})
        
        # 2. 调用 API (传入*完整*历史)
        assistant_response = ask_deepseek_conversational(messages, model)
        
        # 3. 将 LLM 的响应添加到历史 (保持记忆)
        messages.append({"role": "assistant", "content": assistant_response})
        
        # 4. 计入最终的文本文档 (用于输出)
        all_text.append(f"{header_text}\n" + assistant_response)
        
        return assistant_response

    # --- 开始执行4个阶段的*对话* ---

    # 阶段 1：解释概念
    prompt1 = f"""Explain the following two scientific concepts in one concise sentence each:
1. {concept1}
2. {concept2}"""
    run_step(prompt1, "### Step 1: Concept Definitions ###")
    
    # 阶段 2：三轮 (A)-(C)
    for round_id in range(1, 4):
        # 注意：Prompt 中增加了 "Based on our conversation so far" 来强化上下文
        prompt2 = f"""
Based on our conversation so far, let's continue exploring the intersection of “{concept1}” and “{concept2}”.
Round {round_id}:
A) Describe 4 interesting and new scientific contexts in which those two concepts might appear together in a natural and useful way.
B) Criticize the 4 contexts (one short sentence each), based on how well the contexts merge the ideas.
C) Give a 2-sentence reflection summarizing how well these concepts can combine naturally and interestingly."""
        run_step(prompt2, f"\n### Step 2 Round {round_id} (A–C) ###")
    
    # 阶段 3：定义项目标题与目标
    # 提示词也明确要求它"基于之前的所有反思"
    prompt3 = f"""
Excellent. Based on all your previous reflections combining “{concept1}” and “{concept2}”, propose:
- A project title
- A brief explanation (2–3 sentences) of the project's main objective."""
    run_step(prompt3, "### Step 3: Project Proposal ###")
    
    # 阶段 4：研究问题
    # 提示词要求它"基于你刚才提出的项目"
    prompt4 = f"""
Given the project you just proposed, list 2 specific, interesting research questions that would lead to novel and innovative results. 
Each question should be one concise sentence."""
    run_step(prompt4, "### Step 4: Research Questions ###")
    
    # 返回组合好的完整文本文档
    return "\n".join(all_text)


# ========== 主执行函数 (用于批量处理 DataFrame) ==========
def process_concept_pairs_from_df(
    pairs_df: pd.DataFrame, 
    model_name: str = "deepseek-r1:7b", 
    max_items: int = 5,
    output_dir: str = "project_ideas"
):
    """
    (此函数功能不变，但现在调用 *有记忆* 的生成器)
    
    接收概念对 DataFrame，循环生成项目构想，并将结果写入带时间戳的文件。
    """
    
    if pairs_df.empty:
        print("输入 DataFrame 为空，停止生成。", file=sys.stderr)
        return

    df_to_process = pairs_df.head(max_items)
    
    print(f"--- 开始为前 {len(df_to_process)} 个概念对生成项目构想 (DataFrame 模式) ---")
    print(f"使用的模型: {model_name}")
    
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Generating Ideas"):
        c1 = str(row['concept1'])
        c2 = str(row['concept2'])

        print(f"\n=== Generating project idea for: {c1} + {c2} ===\n")
        
        # *** 核心变化：调用新的、有状态的(有记忆的)函数 ***
        idea_text = generate_project_idea_stateful(c1, c2, model=model_name)
        
        print(idea_text)
        print("\n" + "="*80)

        results.append({
            "concept1": c1,
            "concept2": c2,
            "project_idea": idea_text
        })
        
    result_df = pd.DataFrame(results)

    current_time = datetime.now().strftime("%Y%m%d%H%M")
    output_file = os.path.join(output_dir, f"idea_multistage_df_{current_time}.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in result_df.iterrows():
            f.write(f"--- Concept Pair ---\n")
            f.write(f"Concept1: {row['concept1']}\n")
            f.write(f"Concept2: {row['concept2']}\n")
            f.write(f"\n--- LLM Response (Conversational) ---\n")
            f.write(f"{row['project_idea']}\n")
            f.write("\n" + "="*80 + "\n")

    print(f"\n✅ 批量结果已生成并保存到 {output_file}")
    return result_df

# ========== 新增：用于单独运行两个关键词的函数 ==========
def run_single_pair_and_save(
    concept1: str, 
    concept2: str, 
    model: str = "deepseek-r1:7b", 
    output_dir: str = "project_ideas"
):
    """
    这是一个新的辅助函数，完全符合你的要求：
    只传入两个关键词，然后运行并保存结果。
    """
    print(f"\n=== (单个任务) 开始生成: {concept1} + {concept2} ===\n")
    
    # 1. 调用新的 *有状态* 生成函数
    idea_text = generate_project_idea_stateful(concept1, concept2, model=model)
    
    # 2. 打印到控制台
    print(idea_text)
    print("\n" + "="*80)
    
    # 3. 写入文件
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    # 清理文件名
    c1_file = concept1.replace(' ', '_').lower()
    c2_file = concept2.replace(' ', '_').lower()
    output_file = os.path.join(output_dir, f"idea_{c1_file}_{c2_file}_{current_time}.txt")
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"--- Concept Pair ---\n")
            f.write(f"Concept1: {concept1}\n")
            f.write(f"Concept2: {concept2}\n")
            f.write(f"\n--- LLM Response (Conversational) ---\n")
            f.write(f"{idea_text}\n")
        print(f"✅ 单个结果已保存到 {output_file}")
    except Exception as e:
        print(f"Error writing file: {e}", file=sys.stderr)

# import ollama
# import pandas as pd
# from tqdm import tqdm
# from datetime import datetime
# import os
# import sys

# # ========== 统一调用函数 ==========
# def ask_deepseek(prompt, model="deepseek-r1:7b"):
#     """
#     统一封装 Ollama API 调用，处理模型名称和异常。
#     """
#     try:
#         response = ollama.chat(
#             model=model,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response["message"]["content"]
#     except Exception as e:
#         return f"[Error: {e}]"

# # ========== 多阶段生成核心逻辑 ==========
# def generate_project_idea(concept1, concept2, model="deepseek-r1:7b"):
#     """
#     执行四阶段 LLM 调用，为单个概念对生成详细的项目构想。
#     """
#     all_text = []
    
#     # 阶段 1：解释概念
#     prompt1 = f"""Explain the following two scientific concepts in one concise sentence each:
# 1. {concept1}
# 2. {concept2}"""
#     definition_text = ask_deepseek(prompt1, model)
#     all_text.append("### Step 1: Concept Definitions ###\n" + definition_text)
    
#     # 阶段 2：三轮 (A)-(C)
#     for round_id in range(1, 4):
#         prompt2 = f"""
# We are exploring the intersection of “{concept1}” and “{concept2}”.
# Round {round_id}:
# A) Describe 4 interesting and new scientific contexts in which those two concepts might appear together in a natural and useful way.
# B) Criticize the 4 contexts (one short sentence each), based on how well the contexts merge the ideas.
# C) Give a 2-sentence reflection summarizing how well these concepts can combine naturally and interestingly."""
#         round_text = ask_deepseek(prompt2, model)
#         all_text.append(f"\n### Step 2 Round {round_id} (A–C) ###\n" + round_text)
    
#     # 阶段 3：定义项目标题与目标
#     prompt3 = f"""
# Based on your previous reflections combining “{concept1}” and “{concept2}”, propose:
# - A project title
# - A brief explanation (2–3 sentences) of the project's main objective."""
#     project_text = ask_deepseek(prompt3, model)
#     all_text.append("### Step 3: Project Proposal ###\n" + project_text)
    
#     # 阶段 4：研究问题
#     prompt4 = f"""
# Given the project titled above, list 2 specific, interesting research questions that would lead to novel and innovative results. 
# Each question should be one concise sentence."""
#     research_qs = ask_deepseek(prompt4, model)
#     all_text.append("### Step 4: Research Questions ###\n" + research_qs)
    
#     return "\n".join(all_text)


# # ========== 主执行函数 (包含文件写入) ==========
# def process_concept_pairs(
#     pairs_df: pd.DataFrame, 
#     model_name: str = "deepseek-r1:7b", 
#     max_items: int = 5,
#     output_dir: str = "project_ideas"
# ):
#     """
#     接收概念对 DataFrame，循环生成项目构想，并将结果写入带时间戳的文件。

#     Args:
#         pairs_df: 包含 'concept1' 和 'concept2' 列的 DataFrame。
#         model_name: 使用的 Ollama 模型名称。
#         max_items: 限制处理的项目数量。
#         output_dir: 结果文件保存的目录。
#     """
    
#     if pairs_df.empty:
#         print("输入 DataFrame 为空，停止生成。", file=sys.stderr)
#         return

#     # 限制处理的行数
#     df_to_process = pairs_df.head(max_items)
    
#     print(f"--- 开始为前 {len(df_to_process)} 个概念对生成项目构想 ---")
#     print(f"使用的模型: {model_name}")
    
#     results = []
    
#     # 确保输出文件夹存在
#     os.makedirs(output_dir, exist_ok=True)

#     for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Generating Ideas"):
#         concept1, concept2 = row['concept1'], row['concept2']

#         # 确保概念被转换为字符串
#         c1 = str(concept1)
#         c2 = str(concept2)

#         print(f"\n=== Generating project idea for: {c1} + {c2} ===\n")
        
#         # 调用多阶段生成核心逻辑
#         idea_text = generate_project_idea(c1, c2, model=model_name)
        
#         # 打印当前结果
#         print(idea_text)
#         print("\n" + "="*80)

#         results.append({
#             "concept1": c1,
#             "concept2": c2,
#             "project_idea": idea_text
#         })
        
#     # 将结果转换为 DataFrame
#     result_df = pd.DataFrame(results)

#     # 写入文件
#     current_time = datetime.now().strftime("%Y%m%d%H%M")
#     output_file = os.path.join(output_dir, f"idea_multistage_{current_time}.txt")

#     with open(output_file, "w", encoding="utf-8") as f:
#         for _, row in result_df.iterrows():
#             f.write(f"--- Concept Pair ---\n")
#             f.write(f"Concept1: {row['concept1']}\n")
#             f.write(f"Concept2: {row['concept2']}\n")
#             f.write(f"\n--- LLM Response (Multi-Stage) ---\n")
#             f.write(f"{row['project_idea']}\n")
#             f.write("\n" + "="*80 + "\n")

#     print(f"\n✅ 所有构想已生成并保存到 {output_file}")
#     return result_df