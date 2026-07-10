# Entity-Impact-Prediction

**From Keywords to Scientific Entities: Forecasting the Future Impact of Fine-Grained Entity Combinations**

This repository contains the complete source code and datasets for predicting the future scientific impact of fine-grained entity combinations extracted from academic papers in computer science.

---

## Repository Structure

```
Entity-Impact-Prediction/
├── README.md
├── .gitignore
├── src/
│   ├── hgere/                         # HGERE: Entity & Relation Extraction (adapted)
│   │   ├── run_acener.py              # ACE-style NER training script
│   │   ├── run_hgnn.py                # Hypergraph neural network for relation extraction
│   │   ├── run_pruner.py              # Span pruner training script
│   │   ├── utils/
│   │   │   └── data.py                # Data loading utilities
│   │   └── shells/                    # Training shell scripts for ACE04/ACE05/SciERC
│   │
│   ├── impact4cast/                   # Impact4cast: Impact prediction pipeline (adapted)
│   │   ├── features_utils.py          # Feature extraction utilities
│   │   ├── generals_utils.py          # General utility functions
│   │   ├── generate_datasets.py       # Dataset generation script
│   │   ├── idea_generation_utils.py   # Idea/concept generation helpers
│   │   ├── train_model_2019_run_transformer.py   # Train on 2019 data
│   │   ├── train_model_2022_run_transformer.py   # Train on 2022 data
│   │   ├── train_model_utils_transformer.py      # Transformer training utilities
│   │   ├── concept_folder/            # Concept extraction helpers
│   │   ├── create_dynamic_concepts/   # Dynamic concept creation scripts
│   │   ├── create_dynamic_edges/      # Dynamic edge creation & concept pair merging
│   │   ├── domain_concept/            # Domain concept extraction pipeline
│   │   ├── Entities_Corpus/           # Entity corpus preparation
│   │   ├── prepare_eval_data/         # Evaluation data preparation scripts
│   │   └── prepare_other_data/        # Adjacency, pagerank, and prediction utilities
│   │
│   └── analysis/                      # Custom analysis and visualization code
│       ├── entity_cleaning_and_category_score_distribution.ipynb
│       └── all_entity_pair_score_distribution.ipynb
│
└── data/
    ├── scinlp/                        # SciNLP dataset (EMNLP 2025)
    │   ├── train.json                 # Training set
    │   ├── dev.json                   # Development set
    │   └── test.json                  # Test set
    │
    └── entities/                      # Extracted entity data
        ├── extracted_entities_cleaned_final.csv   # Cleaned entity annotations
        └── top10_with_keywords.csv                # Top-10 prediction results with keywords
```

---

## File Descriptions

### src/hgere/ - Entity & Relation Extraction

| File | Description |
|------|-------------|
| run_acener.py | Trains the ACE-style named entity recognition model with span-based entity markers |
| run_hgnn.py | Trains the hypergraph neural network for joint entity and relation extraction |
| run_pruner.py | Trains the span pruning model to filter candidate entity spans |
| utils/data.py | Data loading, preprocessing, and batching utilities for NER/RE tasks |
| shells/ | Bash scripts for training on ACE04, ACE05, and SciERC datasets |

### src/impact4cast/ - Impact Prediction Pipeline

| File | Description |
|------|-------------|
| features_utils.py | Feature engineering: temporal, structural, and semantic features |
| train_model_utils_transformer.py | Transformer-based model training with checkpoint resume |
| train_model_2019_run_transformer.py | Training script using 2019 publication data |
| train_model_2022_run_transformer.py | Training script using 2022 publication data |
| generate_datasets.py | Generates training/evaluation datasets from raw concept data |
| idea_generation_utils.py | Utilities for generating concept combinations and candidate pairs |
| create_dynamic_concepts/ | Scripts for dynamic concept extraction and citation filtering |
| create_dynamic_edges/ | Scripts for concept pair generation and edge construction |
| domain_concept/ | Notebooks for domain-specific concept extraction from CS/CL papers |
| prepare_eval_data/ | Scripts for preparing feature-based evaluation datasets |
| prepare_other_data/ | Adjacency matrix, PageRank, and future link prediction utilities |

### src/analysis/ - Custom Analysis & Visualization

| File | Description |
|------|-------------|
| entity_cleaning_and_category_score_distribution.ipynb | Entity deduplication, cleaning, and score distribution analysis by entity category |
| all_entity_pair_score_distribution.ipynb | Comprehensive analysis of entity pair score distributions and statistics |

### data/scinlp/ - SciNLP Dataset

| File | Size | Description |
|------|------|-------------|
| train.json | 2.54 MB | Training documents with entity and relation annotations |
| dev.json | 0.30 MB | Development documents |
| test.json | 0.20 MB | Test documents |

### data/entities/ - Extracted Entity Data

| File | Description |
|------|-------------|
| extracted_entities_cleaned_final.csv | Cleaned fine-grained scientific entity annotations |
| top10_with_keywords.csv | Top-10 predicted entity pairs with scores and keywords |

**CSV Schema (extracted_entities_cleaned_final.csv):**

| Column | Type | Description |
|--------|------|-------------|
| doc_id | string | Document identifier |
| entity_name | string | Extracted entity surface form |
| entity_type | string | Fine-grained entity type (e.g., method, task, dataset, metric) |
| start_pos | int | Character start position in document |
| end_pos | int | Character end position in document |

---

## Data Attribution & Sources

### Original Code Repositories

This project builds upon the following open-source codebases. Each source code file includes an attribution header at the top indicating its origin and modifications made.

1. **HGERE** - "Joint Entity and Relation Extraction with Span Pruning and Hypergraph Neural Networks" (EMNLP 2023)
   - Repository: https://github.com/yanzhh/HGERE
   - Based on: PL-Marker (https://github.com/thunlp/PL-Marker)
   - Modifications: Bug fixes for checkpoint resume mechanism; dataset adaptation for scientific entity impact prediction.

2. **Impact4cast** - Impact prediction framework from the Max Planck Institute for Informatics
   - Original authors: Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy
   - Modifications: Bug fixes for checkpoint resume; dataset replacement with SciNLP dataset.

3. **SciNLP** - "SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP" (EMNLP 2025)
   - Repository: https://github.com/AKADDC/SciNLP
   - Dataset based on: The ACL OCL Corpus (https://github.com/shauryr/ACL-anthology-corpus)

### Custom Code

The src/analysis/ directory contains our own analysis and visualization code developed specifically for this project.

---

## Citation

If you use this code or dataset in your research, please cite:

```
@article{entity-impact-prediction,
  title     = {From Keywords to Scientific Entities: Forecasting the Future Impact of Fine-Grained Entity Combinations},
  journal   = {Under Review},
  year      = {2025}
}
```

And the original works this project builds upon:

```
@inproceedings{yan2023hgere,
  title     = {Joint Entity and Relation Extraction with Span Pruning and Hypergraph Neural Networks},
  author    = {Yan, Zhaohui and others},
  booktitle = {Proceedings of EMNLP},
  year      = {2023}
}

@inproceedings{scinlp2025,
  title     = {SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP},
  author    = {AKADDC},
  booktitle = {Proceedings of EMNLP},
  year      = {2025}
}
```

---

## Setup & Usage

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers (HuggingFace)
- Jupyter Notebook (for analysis scripts)

### Installation

```bash
pip install torch transformers pandas numpy matplotlib seaborn scipy scikit-learn
```

### Entity Extraction (HGERE)

```bash
# Train pruner
bash src/hgere/shells/pruner/scierc/run_train_pruner_scierc.sh

# Train extraction model
bash src/hgere/shells/hgere/scierc/run_train_scierc_scibert_hgere.sh
```

### Impact Prediction (Impact4cast)

```bash
# Generate datasets
python src/impact4cast/generate_datasets.py

# Train prediction model
python src/impact4cast/train_model_2022_run_transformer.py
```

### Analysis

Open the Jupyter notebooks in src/analysis/ to reproduce the analysis figures and statistics.

---

---

## Notes

- Large intermediate data files (model checkpoints, raw citation data in .gz format, KG_triples.txt, and large prediction CSV files) have been excluded to keep the repository size under GitHub's limit.
- All source code files include attribution headers indicating their original source and modifications.
- All comments, variable names, and data headers are in English.
- For the full raw data or model checkpoints, please refer to the original repositories linked above.
