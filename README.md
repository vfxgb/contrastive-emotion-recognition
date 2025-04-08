# 🎭 Contrastive-Mamba Emotion Recognition (EmoCon)
**Enhancing Text Emotion Recognition with Mamba and Contrastive Learning for Robust Emotion Embeddings**

---

## 📌 Overview
This project integrates the powerful **Selective State Space Model (Mamba)** with **Contrastive Learning** to enhance **Text Emotion Recognition (TER)**.  
Traditional models (e.g., CNN, RNN, BERT) often:
- ❌ **Fail to capture global sentence context effectively**.
- ❌ **Struggle with cross-dataset generalization and unseen emotions**.

We propose an innovative **Mamba-Contrastive approach** that:
- Captures both **local (word-level)** and **global (sentence-level)** emotional contexts using **Selective Mamba**.
- Learns robust, generalizable emotion-aware embeddings through **Contrastive Learning**, enhancing performance on new, unseen data.

---

## 🎯 Research Question
> How can integrating **Selective State Space Models (Mamba)** with **Contrastive Learning** significantly improve **emotion embedding quality**, **domain generalization**, and robustness to **unseen emotions**?

---

## 🚀 Key Features
✅ **Baseline Model:** Fine-tuned BERT for standard emotion classification.  
✅ **Proposed Model:** **Selective Mamba + Contrastive Learning** to leverage global context and learn robust emotion embeddings.  
✅ **Cross-Dataset Evaluation:** Train on **CrowdFlower**, evaluate on **WASSA 2021** & **ISEAR** to test generalization.  

---

## 📂 Updated Repository Structure - To DO
```plaintext
contrastive-emotion-recognition/
│── data/                          # Datasets (CrowdFlower, WASSA)
│── notebooks/                     # Jupyter Notebooks (visualizations, analysis)
│── src/
│   ├── models/
│   │   ├── mamba.py               # Selective Mamba implementation
│   │   └── contrastive_model.py   # Integrated Mamba-Contrastive model
│   ├── preprocess_data.py         # Data preprocessing
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   ├── contrastive_loss.py        # Contrastive loss implementation
│   └── utils.py                   # Utility functions
│── results/                       # Experimental results and visualizations
│── README.md                      # Project documentation
│── requirements.txt               # Dependencies
│── report/                        # Final project report
```

---

## 📊 Datasets Used
- **CrowdFlower Emotion Dataset** (Training)
- **WASSA 2021** (Domain generalization testing)
- **ISEAR** (Domain generalization testing)

---

## ⚡ Installation & Setup
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/contrastive-emotion-recognition.git
cd contrastive-emotion-recognition.git
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


**Examples**:
```bash
# Process all datasets (preprocess + train/test)
./run_pipeline.sh --all

# Preprocess WASSA only
./run_pipeline.sh --preprocess wassa

# Train/test on ISEAR and CrowdFlower
./run_pipeline.sh --train_test isear
./run_pipeline.sh --train_test crowdflower
```

## 🛠️ Model Architectures
- **Baseline:** Fine-tuned BERT classifier.
- **Proposed:** **Selective Mamba** embeddings optimized with **Supervised + Self-Supervised Contrastive Learning** for enhanced emotion discrimination.
- **Metrics:** Accuracy, F1-score, and t-SNE visualizations.

---

## 📌 Expected Results - TBC
✅ **Hypothesis:** The integration of Mamba and Contrastive Learning will yield superior performance and better generalization.

| **Model**                   | **Dataset**     | **Accuracy** | **F1-Score** |
|-----------------------------|-----------------|--------------|--------------|
| Bilstm                      | CrowdFlower     | 85.2%        | 84.8%        |
| Bilstm                      | WASSA           | 78.5%        | 77.9%        |
| Bilstm                      | ISEAR           | 78.5%        | 77.9%        |
| **Contrastive-Mamba (ours)**| **CrowdFlower** | **89.0%**    | **88.5%**    |
| **Contrastive-Mamba (ours)**| **WASSA**       | **85.0%**    | **84.3%**    |
| **Contrastive-Mamba (ours)**| **ISEAR**       | **85.0%**    | **84.3%**    |

---

## 📌 To-Do Checklist
- [x] Load and preprocess datasets
- [x] Implement Baseline BERT classifier
- [x] Implement Contrastive Loss
- [x] Implement Mamba Model
- [ ] Fine-tune Contrastive-Mamba Model
- [ ] Evaluate cross-dataset generalization
- [ ] Few-shot evaluation
- [ ] Write and submit final report

---

## 🤝 Contributing
Feel free to contribute! Fork the repo, create a new branch, and submit a pull request.

---

## 📜 References
- Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
- Gao, T., Yao, X., & Chen, D. (2021). "SimCSE: Simple Contrastive Learning of Sentence Embeddings".
- Gunel, B., et al. (2021). "Supervised Contrastive Learning for Pretrained Language Model Fine-Tuning".
- CrowdFlower Dataset: [Link](https://data.world/crowdflower/sentiment-analysis-in-text)
- WASSA 2017 Dataset: [Link](https://github.com/vinayakumarr/WASSA-2017)

---

## 🏆 Acknowledgments
This project is part of SC4001 CE/CZ4042: Neural Networks and Deep Learning.

