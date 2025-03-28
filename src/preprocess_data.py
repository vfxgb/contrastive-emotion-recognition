import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Subset
import re
import numpy as np
import os
import spacy
import string
from collections import Counter

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
nlp = spacy.load("en_core_web_sm")

# Clean tweets/text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Load and process CrowdFlower dataset
def load_crowdflower(path, max_length=128):
    df = pd.read_csv(path)
    print(f"[CrowdFlower] Loaded {len(df)} rows from {path}")

    df['original'] = df['content']  # Keep original for debugging
    df['content'] = df['content'].apply(clean_text)

    # Print a few samples
    print("\n[SAMPLE CLEANED DATA]")
    print(df[['original', 'content', 'sentiment']].sample(5))

    # Encode labels
    unique_sentiments = sorted(df['sentiment'].unique())
    label_map = {label: idx for idx, label in enumerate(unique_sentiments)}
    print(f"\n[CrowdFlower] Label mapping: {label_map}")

    labels = df['sentiment'].map(label_map).tolist()
    label_counts = Counter(labels)
    print(f"[Label Distribution BEFORE split]: {label_counts}")

    texts = df['content'].tolist()
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    # Debug: avg length
    input_lengths = [len(tokenizer.tokenize(t)) for t in texts]
    print(f"\n[Stats] Avg token length: {np.mean(input_lengths):.2f} | Max: {np.max(input_lengths)}")

    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    label_tensor = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, label_tensor)
    print(f"\n[CrowdFlower] Final dataset shape: {len(dataset)} samples")

    return dataset, texts, labels, label_map

# Proper splitting using Subset
def split_dataset(dataset, split_ratio=0.8, seed=42):
    from torch.utils.data.dataset import Subset
    from sklearn.model_selection import train_test_split

    input_ids = dataset.tensors[0]
    attention_mask = dataset.tensors[1]
    labels = dataset.tensors[2]

    # Convert to list of text tokens to ensure no overlaps
    text_ids = [tuple(row.tolist()) for row in input_ids]  # immutable for hashing
    unique_texts, indices = np.unique(text_ids, return_index=True, axis=0)

    X_train_idx, X_test_idx = train_test_split(
        indices,
        train_size=split_ratio,
        random_state=seed,
        stratify=labels[indices].numpy()  # Stratified sampling to maintain label dist
    )

    train_ds = Subset(dataset, X_train_idx)
    test_ds = Subset(dataset, X_test_idx)

    print(f"[Split] Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    return train_ds, test_ds


if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    
    print("[Main] Loading and processing CrowdFlower dataset...")
    full_dataset, all_texts, all_labels, label_map = load_crowdflower('../data/CrowdFlower/text_emotion.csv')

    print("[Main] Splitting dataset into train and test...")
    train_ds, test_ds = split_dataset(full_dataset, split_ratio=0.8)
    # Extract unique input_ids (as tuples) for train and test subsets
    train_ids = set([tuple(train_ds.dataset.tensors[0][i].tolist()) for i in train_ds.indices])
    test_ids  = set([tuple(test_ds.dataset.tensors[0][i].tolist()) for i in test_ds.indices])

    # Compute the intersection of train and test sets
    overlap = train_ids.intersection(test_ids)
    print(f"[Leakage Check] Overlapping samples between train & test: {len(overlap)}")

    print("[Main] Saving datasets to disk...")
    torch.save(train_ds, '../data/train.pt')
    torch.save(test_ds, '../data/test.pt')
    print("[Main] Done.")
