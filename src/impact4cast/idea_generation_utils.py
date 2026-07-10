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

# ========== Unified call function (core modification) ==========
def ask_deepseek_conversational(messages, model="deepseek-r1:7b"):
    """
    Unified wrapper for Ollama API calls (modified).
    
     ...  * ... * (messages )
    and sends it to Ollama.
    It returns the assistant's new response content.
    
    Args:
        messages (list):  Ollama/OpenAI  ... 
        model (str):  ... 
        
    Returns:
        str:  ... 
    """
    try:
        response = ollama.chat(
            model=model,
            messages=messages  #  ... 
        )
        #  ...  **  ... 
        return response["message"]["content"]
    except Exception as e:
        return f"[Error: {e}]"

# ==========  ... ** ...  ( ... ) ==========
def generate_project_idea_stateful(concept1, concept2, model="deepseek-r1:7b"):
    """
     ...  LLM ** ( ... ) ... 
    This maintains a unified conversation history.
    
    Args:
        concept1 (str):  ... 
        concept2 (str):  ... 
        model (str):  ... 
        
    Returns:
        str:  ... 
    """
    
    # 1.  ...  ()
    #  ... 
    messages = []
    
    # 2.  ... 
    # ( ... )
    all_text = []

    # --- Internal helper function to run a dialogue step ---
    def run_step(prompt_text, header_text):
        """
         ... 
        1. user's prompt to history
        2. call API to get response
        3. assistant's response to history
        4.  ... 
        """
        nonlocal messages, all_text
        
        # 1. user's new prompt to history
        messages.append({"role": "user", "content": prompt_text})
        
        # 2.  API (**)
        assistant_response = ask_deepseek_conversational(messages, model)
        
        # 3.  LLM response to history (maintain memory)
        messages.append({"role": "assistant", "content": assistant_response})
        
        # 4.  ...  ( ... )
        all_text.append(f"{header_text}\n" + assistant_response)
        
        return assistant_response

    # ---  ... 4 ... ** ---

    # Stage 1: Explain concepts
    prompt1 = f"""Explain the following two scientific concepts in one concise sentence each:
1. {concept1}
2. {concept2}"""
    run_step(prompt1, "### Step 1: Concept Definitions ###")
    
    # Stage 2: Three rounds (A)-(C)
    for round_id in range(1, 4):
        #  ... Prompt  ...  "Based on our conversation so far"  ... 
        prompt2 = f"""
Based on our conversation so far, let's continue exploring the intersection of {concept1} and {concept2}.
Round {round_id}:
A) Describe 4 interesting and new scientific contexts in which those two concepts might appear together in a natural and useful way.
B) Criticize the 4 contexts (one short sentence each), based on how well the contexts merge the ideas.
C) Give a 2-sentence reflection summarizing how well these concepts can combine naturally and interestingly."""
        run_step(prompt2, f"\n### Step 2 Round {round_id} (AC) ###")
    
    # Stage 3: Define project title and objective
    #  ... " ... "
    prompt3 = f"""
Excellent. Based on all your previous reflections combining {concept1} and {concept2}, propose:
- A project title
- A brief explanation (23 sentences) of the project's main objective."""
    run_step(prompt3, "### Step 3: Project Proposal ###")
    
    # Stage 4: Research questions
    #  ... " ... "
    prompt4 = f"""
Given the project you just proposed, list 2 specific, interesting research questions that would lead to novel and innovative results. 
Each question should be one concise sentence."""
    run_step(prompt4, "### Step 4: Research Questions ###")
    
    #  ... 
    return "\n".join(all_text)


# ==========  ...  ( ...  DataFrame) ==========
def process_concept_pairs_from_df(
    pairs_df: pd.DataFrame, 
    model_name: str = "deepseek-r1:7b", 
    max_items: int = 5,
    output_dir: str = "project_ideas"
):
    """
    ( ...  * ... *  ... )
    
     ...  DataFrame ... 
    """
    
    if pairs_df.empty:
        print(" DataFrame  ... ", file=sys.stderr)
        return

    df_to_process = pairs_df.head(max_items)
    
    print(f"---  ...  {len(df_to_process)}  ...  (DataFrame ) ---")
    print(f" ... : {model_name}")
    
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Generating Ideas"):
        c1 = str(row['concept1'])
        c2 = str(row['concept2'])

        print(f"\n=== Generating project idea for: {c1} + {c2} ===\n")
        
        # ***  ... ( ... ) ***
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

    print(f"\n✅  ...  {output_file}")
    return result_df

# ==========  ...  ==========
def run_single_pair_and_save(
    concept1: str, 
    concept2: str, 
    model: str = "deepseek-r1:7b", 
    output_dir: str = "project_ideas"
):
    """
     ... 
     ... 
    """
    print(f"\n=== ( ... )  ... : {concept1} + {concept2} ===\n")
    
    # 1.  ...  * ... *  ... 
    idea_text = generate_project_idea_stateful(concept1, concept2, model=model)
    
    # 2.  ... 
    print(idea_text)
    print("\n" + "="*80)
    
    # 3.  ... 
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    #  ... 
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
        print(f"✅  ...  {output_file}")
    except Exception as e:
        print(f"Error writing file: {e}", file=sys.stderr)

# import ollama
# import pandas as pd
# from tqdm import tqdm
# from datetime import datetime
# import os
# import sys

# # ==========  ...  ==========
# def ask_deepseek(prompt, model="deepseek-r1:7b"):
#     """
#      ...  Ollama API  ... 
#     """
#     try:
#         response = ollama.chat(
#             model=model,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response["message"]["content"]
#     except Exception as e:
#         return f"[Error: {e}]"

# # ==========  ...  ==========
# def generate_project_idea(concept1, concept2, model="deepseek-r1:7b"):
#     """
#      ...  LLM  ... 
#     """
#     all_text = []
    
#     # Stage 1: Explain concepts
#     prompt1 = f"""Explain the following two scientific concepts in one concise sentence each:
# 1. {concept1}
# 2. {concept2}"""
#     definition_text = ask_deepseek(prompt1, model)
#     all_text.append("### Step 1: Concept Definitions ###\n" + definition_text)
    
#     # Stage 2: Three rounds (A)-(C)
#     for round_id in range(1, 4):
#         prompt2 = f"""
# We are exploring the intersection of {concept1} and {concept2}.
# Round {round_id}:
# A) Describe 4 interesting and new scientific contexts in which those two concepts might appear together in a natural and useful way.
# B) Criticize the 4 contexts (one short sentence each), based on how well the contexts merge the ideas.
# C) Give a 2-sentence reflection summarizing how well these concepts can combine naturally and interestingly."""
#         round_text = ask_deepseek(prompt2, model)
#         all_text.append(f"\n### Step 2 Round {round_id} (AC) ###\n" + round_text)
    
#     # Stage 3: Define project title and objective
#     prompt3 = f"""
# Based on your previous reflections combining {concept1} and {concept2}, propose:
# - A project title
# - A brief explanation (23 sentences) of the project's main objective."""
#     project_text = ask_deepseek(prompt3, model)
#     all_text.append("### Step 3: Project Proposal ###\n" + project_text)
    
#     # Stage 4: Research questions
#     prompt4 = f"""
# Given the project titled above, list 2 specific, interesting research questions that would lead to novel and innovative results. 
# Each question should be one concise sentence."""
#     research_qs = ask_deepseek(prompt4, model)
#     all_text.append("### Step 4: Research Questions ###\n" + research_qs)
    
#     return "\n".join(all_text)


# # ==========  ...  ( ... ) ==========
# def process_concept_pairs(
#     pairs_df: pd.DataFrame, 
#     model_name: str = "deepseek-r1:7b", 
#     max_items: int = 5,
#     output_dir: str = "project_ideas"
# ):
#     """
#      ...  DataFrame ... 

#     Args:
#         pairs_df:  'concept1'  'concept2'  DataFrame
#         model_name:  ...  Ollama  ... 
#         max_items:  ... 
#         output_dir:  ... 
#     """
    
#     if pairs_df.empty:
#         print(" DataFrame  ... ", file=sys.stderr)
#         return

#     #  ... 
#     df_to_process = pairs_df.head(max_items)
    
#     print(f"---  ...  {len(df_to_process)}  ...  ---")
#     print(f" ... : {model_name}")
    
#     results = []
    
#     # Output file ... 
#     os.makedirs(output_dir, exist_ok=True)

#     for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Generating Ideas"):
#         concept1, concept2 = row['concept1'], row['concept2']

#         #  ... 
#         c1 = str(concept1)
#         c2 = str(concept2)

#         print(f"\n=== Generating project idea for: {c1} + {c2} ===\n")
        
#         #  ... 
#         idea_text = generate_project_idea(c1, c2, model=model_name)
        
#         #  ... 
#         print(idea_text)
#         print("\n" + "="*80)

#         results.append({
#             "concept1": c1,
#             "concept2": c2,
#             "project_idea": idea_text
#         })
        
#     #  ...  DataFrame
#     result_df = pd.DataFrame(results)

#     #  ... 
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

#     print(f"\n✅  ...  {output_file}")
#     return result_df