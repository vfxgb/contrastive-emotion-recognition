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
import random

# Initialize tokenizer and spaCy model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
nlp = spacy.load("en_core_web_sm")

# Clean tweets/text
def clean_text(text):
    # Remove URLs, mentions, hashtags, and HTML entities
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    text = re.sub(r'&amp;', '&', text)

    return text  

def load_crowdflower(path, max_length=128, min_samples=1000):
    df = pd.read_csv(path)
    print(f"[CrowdFlower] Loaded {len(df)} rows from {path}")

    df['original'] = df['content']  # Keep original for debugging
    df['content'] = df['content'].apply(clean_text)

    # Encode labels
    unique_sentiments = sorted(df['sentiment'].unique())
    label_map = {label: idx for idx, label in enumerate(unique_sentiments)}
    df['label'] = df['sentiment'].map(label_map)

    # Print original distribution
    print(f"[Original Label Distribution]")
    print(df['label'].value_counts())

    # Filter out classes with < min_samples
    class_counts = df['label'].value_counts()
    keep_classes = class_counts[class_counts >= min_samples].index.tolist()
    df_filtered = df[df['label'].isin(keep_classes)].reset_index(drop=True)

    print(f"[Filtered Label Distribution (min {min_samples} per class)]")
    print(df_filtered['label'].value_counts())
    print("\n[Filtered Sentiment Distribution]")
    print(df_filtered['sentiment'].value_counts())

    # Re-map labels to be consecutive
    new_label_map = {old: i for i, old in enumerate(sorted(df_filtered['label'].unique()))}
    df_filtered['label'] = df_filtered['label'].map(new_label_map)

    texts = df_filtered['content'].tolist()
    labels = df_filtered['label'].tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    label_tensor = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, label_tensor)
    print(f"[CrowdFlower] Final FILTERED dataset shape: {len(dataset)} samples")
    print(f"[Final Label Mapping] {new_label_map}")

    return dataset, texts, labels, new_label_map

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
        stratify=labels[indices].numpy()  # Stratified sampling to maintain label distribution
    )

    train_ds = Subset(dataset, X_train_idx)
    test_ds = Subset(dataset, X_test_idx)

    print(f"[Split] Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    return train_ds, test_ds

# --- New: DualViewDataset for Contrastive Learning ---

def random_dropout_tokens(token_ids, dropout_prob=0.1):
    """
    Simple augmentation: randomly drop tokens (except special tokens).
    Assumes special tokens: [CLS]=101, [SEP]=102, [PAD]=0.
    """
    return [tok for tok in token_ids if random.random() > dropout_prob or tok in [101, 102, 0]]

class DualViewDataset(torch.utils.data.Dataset):
    def __init__(self, subset, dropout_prob=0.1):
        """
        Handles both TensorDataset and Subset objects
        """
        if isinstance(subset, torch.utils.data.Subset):
            self.dataset = subset.dataset
            self.indices = subset.indices
        else:
            self.dataset = subset
            self.indices = list(range(len(subset)))
            
        self.dropout_prob = dropout_prob

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get original sample
        original_idx = self.indices[idx]
        input_ids, attention_mask, label = self.dataset[original_idx]
        
        # Create two augmented views
        view1 = random_dropout_tokens(input_ids.tolist(), self.dropout_prob)
        view2 = random_dropout_tokens(input_ids.tolist(), self.dropout_prob)
        
        # Pad to original length
        max_len = input_ids.size(0)
        view1 = view1 + [0] * (max_len - len(view1))
        view2 = view2 + [0] * (max_len - len(view2))
        
        return torch.tensor(view1), torch.tensor(view2), label

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    print("[Main] Loading and processing CrowdFlower dataset...")
    full_dataset, all_texts, all_labels, label_map = load_crowdflower('data/CrowdFlower/text_emotion.csv')

    print("[Main] Splitting dataset into train and test...")
    train_ds, test_ds = split_dataset(full_dataset, split_ratio=0.8)
    # Extract unique input_ids (as tuples) for train and test subsets
    train_ids = set([tuple(train_ds.dataset.tensors[0][i].tolist()) for i in train_ds.indices])
    test_ids  = set([tuple(test_ds.dataset.tensors[0][i].tolist()) for i in test_ds.indices])

    # Compute the intersection of train and test sets
    overlap = train_ids.intersection(test_ids)
    print(f"[Leakage Check] Overlapping samples between train & test: {len(overlap)}")

    print("[Main] Saving datasets to disk...")
    torch.save(train_ds, 'data/preprocessed_dataset/crowdflower/train.pt')
    torch.save(test_ds, 'data/preprocessed_dataset/crowdflower/test.pt')
    print("[Main] Done.")
