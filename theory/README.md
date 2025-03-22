## 📖 Concepts & Theory

### 1. Text Emotion Recognition (TER)

Text Emotion Recognition involves identifying and classifying emotions expressed in textual data. Traditional methods primarily rely on capturing local contexts (word-level information), often neglecting global sentence-level contexts crucial for accurate emotion recognition.

### 2. Challenges with Traditional Methods

- **Local Context Dependency**: CNNs and RNNs capture only local or sequential patterns, missing broader emotional context.
- **Generalization Issues**: Models trained on one dataset struggle to perform well across different datasets.
- **Few-shot Limitations**: Difficulty recognizing unseen emotions with limited labeled examples.

To address these challenges, we propose integrating **Contrastive Learning** with the **Selective State Space Model (Mamba)**.

---

### 3. Contrastive Learning

Contrastive learning focuses on learning representations by contrasting positive pairs (similar examples) against negative pairs (dissimilar examples). This technique maximizes the similarity of embeddings representing the same emotion and minimizes similarity between embeddings of different emotions.

**Key Idea:**

- Positive pairs: Text samples expressing the same emotion.
- Negative pairs: Text samples expressing different emotions.

**Benefits:**

- Produces embeddings with distinct clusters for each emotion.
- Improves model robustness and generalization capabilities.

#### Supervised + Self-Supervised Contrastive Learning

- **Supervised contrastive learning** uses label information explicitly to define positive and negative pairs.
- **Self-supervised contrastive learning** leverages data augmentation or semantic perturbations of text to create positive pairs.

Combining both allows the model to leverage explicit emotion labels while also learning robust representations from unlabeled or weakly labeled data.

---

### 4. Selective State Space Model (Mamba)

Mamba is a recent deep learning architecture that leverages selective state space representations, capturing long-range dependencies effectively with linear computational complexity.

**Key Components of Mamba:**

- **Selective State Spaces (SSS)**: Dynamically adjusts computations based on input-dependent parameters, capturing both local and global contexts efficiently.
- **Linear-Time Complexity**: Scales linearly with sequence length, enabling efficient modeling of long text sequences compared to traditional transformers.
- **Robust Sequence Modeling**: Effectively models both short- and long-range contextual dependencies, ideal for capturing the nuances of emotional expressions in text.

---

### 5. Integration: Contrastive-Mamba Architecture

The proposed **Contrastive-Mamba** architecture integrates Mamba's powerful sequence modeling with Contrastive Learning's ability to produce discriminative, emotion-aware embeddings:

- **Step 1: Input Embedding with Mamba**

  - Text sequences are tokenized and embedded through a Selective Mamba model.
  - Mamba captures both local (words/phrases) and global (sentence-level) emotional contexts.

- **Step 2: Embedding Projection**

  - Mamba outputs embeddings that are projected into a lower-dimensional embedding space optimized for distinguishing emotions.

- **Step 3: Contrastive Optimization**

  - The embeddings undergo supervised and self-supervised contrastive training, encouraging embeddings of the same emotion to cluster closely and different emotions to separate distinctly.

- **Step 4: Classification Layer**

  - A simple classifier utilizes optimized contrastive embeddings to predict emotion labels effectively.

---

### 6. Why This Works?

- **Global and Local Context Integration**: Mamba effectively encodes both detailed local word context and broad global sentence context crucial for accurate emotion recognition.
- **Robust Embeddings**: Contrastive Learning creates discriminative emotion clusters, enhancing model performance and robustness.
- **Generalization Across Datasets**: The contrastive optimization enhances domain generalization, enabling the model to recognize emotions effectively in different datasets.
- **Few-shot Capability**: The distinct embedding clusters learned via contrastive methods significantly improve few-shot emotion recognition performance.

---

This integrated approach uniquely combines Mamba’s efficient global context capture with contrastive learning’s discriminative embedding optimization, significantly advancing the state-of-the-art in robust and generalized text emotion recognition.

