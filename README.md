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
git clone https://github.com/yourusername/contrastive-emotion-recognition.git
cd contrastive-emotion-recognition
```

### **Step 2: Create a Virtual Environment & Install Dependencies**
```bash
conda env create -f environment.yaml
conda activate SC4001
```
### **Step 3: Run the Pipeline**

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

## Contributing
Feel free to contribute! Fork the repo, create a new branch, and submit a pull request.

---

## References
- Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
- Gao, T., Yao, X., & Chen, D. (2021). "SimCSE: Simple Contrastive Learning of Sentence Embeddings".
- Gunel, B., et al. (2021). "Supervised Contrastive Learning for Pretrained Language Model Fine-Tuning".
- CrowdFlower Dataset: [Link](https://data.world/crowdflower/sentiment-analysis-in-text)
- WASSA 2017 Dataset: [Link](https://github.com/vinayakumarr/WASSA-2017)

---

## Acknowledgments
This project is part of SC4001 CE/CZ4042: Neural Networks and Deep Learning.

