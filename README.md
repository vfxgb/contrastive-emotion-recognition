# 🎭 Contrastive-Mamba Emotion Recognition (EmoCon)
**Enhancing Text Emotion Recognition with Mamba and Contrastive Learning for Robust Emotion Embeddings**

---

## Overview
This project integrates the powerful **Selective State Space Model (Mamba)** with **Contrastive Learning** to enhance **Text Emotion Recognition (TER)**.  
Traditional models (e.g., CNN, RNN, BERT) often:
- ❌ **Fail to capture global sentence context effectively**.
- ❌ **Struggle with cross-dataset generalization and unseen emotions**.

We propose an innovative **Mamba-Contrastive approach** that:
- Captures both **local (word-level)** and **global (sentence-level)** emotional contexts using **Selective Mamba**.
- Learns robust, generalizable emotion-aware embeddings through **Contrastive Learning**, enhancing performance on new, unseen data.

---

## Research Question
> How can integrating **Selective State Space Models (Mamba)** with **Contrastive Learning** significantly improve **emotion embedding quality**, **domain generalization**, and robustness to **unseen emotions**?

---

## Key Features
✅ **Proposed Model:** **Selective Mamba + Contrastive Learning** to leverage global context and learn robust emotion embeddings.  
✅ **Cross-Dataset Evaluation:** Train on **CrowdFlower**, evaluate on **WASSA 2021** & **ISEAR** to test generalization.  

---

## Datasets Used
- **CrowdFlower Emotion Dataset** (Training)
- **WASSA 2021** (Domain generalization testing)
- **ISEAR** (Domain generalization testing)

---

## Installation & Setup
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/vfxgb/contrastive-emotion-recognition.git
cd contrastive-emotion-recognition
```

### **Step 2: Create a Virtual Environment & Install Dependencies**

```bash
conda create -n mamba_contrastive python==3.10.6
conda activate mamba_contrastive
pip install -e .
pip install --no-cache-dir --no-binary causal-conv1d causal-conv1d
```

### **Step 3: Run the Pipeline**
```bash
chmod +x run_pipeline.sh
```

#### If using SLURM Cluster (use this instead for the rest of the commands)
```bash
chmod +x job_run_pipeline.sh 
```


| Command                                                                                        | Description                                                                                                      | Example                                                                                           |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `./run_pipeline.sh <dataset> <model>`                                                          | Runs preprocessing and train/test for the given dataset and model.                                               | `./run_pipeline.sh crowdflower mamba`                                                              |
| `./run_pipeline.sh --force_preprocess <dataset> <model>`                                       | Forces re-preprocessing (even if already done) and then runs train/test.                                         | `./run_pipeline.sh --force_preprocess crowdflower bilstm_glove`                                    |
| `./run_pipeline.sh <dataset> <model> --finetune_mode <1\|2\|3>`                                | Runs with a specified finetuning strategy (only required when dataset is `isear` or `wassa` + model is `mamba` or `bilstm_bert`). | `./run_pipeline.sh isear bilstm_bert --finetune_mode 3`                                            |
| `./run_pipeline.sh --force_preprocess <dataset> <model> --finetune_mode <1\|2\|3>`             | Combines forced re-preprocessing **and** a specified finetuning strategy.                                        | `./run_pipeline.sh --force_preprocess wassa mamba --finetune_mode 1`                              |

**Notes:**
- **Datasets:** `crowdflower`, `isear`, `wassa`
- **Models:** `bilstm_glove`, `bilstm_bert`, `mamba`
- **Finetune modes:**  
  1. Load checkpoint, freeze encoder, finetune classifier  
  2. Load checkpoint, finetune encoder and classifier  
  3. Train completely from scratch  

`--finetune_mode` is **only required** if you're training on `isear` or `wassa` **with** `mamba` or `bilstm_bert`.

`--force_preprocess` is optional; if omitted, the script will only preprocess if it hasn’t been done before.

---

## Datasets
- [CrowdFlower Dataset](https://data.world/crowdflower/sentiment-analysis-in-text)
- [WASSA 2021 Dataset](https://lt3.ugent.be/resources/wassa-2021-shared-task/dataset-download-form/)
- [ISEAR Dataset](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)


---

## Acknowledgments
This project is part of SC4001 CE/CZ4042: Neural Networks and Deep Learning.

