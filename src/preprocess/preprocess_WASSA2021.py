import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split, Subset, Dataset
from transformers import AutoTokenizer
import re
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from utils import clean_text, fetch_label_mapping

# --- Helper Functions ---
label_mapping = fetch_label_mapping(wassa=True)

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_wassa(tsv_path, max_length=128):
    """
    Load and preprocess the WASSA 2021 dataset from a TSV file.
    Renames 'emotion_labels' to 'Emotion' and 'essays' to 'Text'.
    Filters to keep only rows with one of the six Ekman emotions,
    cleans the text, tokenizes it, and maps the labels.
    Returns a TensorDataset.
    """
    print("Loading TSV from:", tsv_path)
    df = pd.read_csv(tsv_path, sep='\t')
    print("Original dataset shape:", df.shape)
    print("Columns in dataset:", df.columns.tolist())
    
    # Rename columns for consistency
    df = df.rename(columns={'emotion_label': 'Emotion', 'essay': 'Text'})
    print("Renamed columns: 'emotion_labels' -> 'Emotion', 'essays' -> 'Text'")
    
    # Show original label distribution
    print("Original label distribution in 'Emotion':")
    print(df['Emotion'].value_counts())
    
    # Filter to keep only rows with desired emotions (exclude others such as 'neutral')
    df = df[df['Emotion'].isin(label_mapping.keys())].reset_index(drop=True)
    print("Filtered dataset shape:", df.shape)
    print("Filtered label distribution:")
    print(df['Emotion'].value_counts())
    
    # Clean the text in the 'Text' column
    df['content'] = df['Text'].apply(clean_text)
    print("Sample cleaned text:", df['content'].iloc[0])
    
    # Tokenize the cleaned text using the BERT tokenizer
    encodings = tokenizer(
        df['content'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    print("Tokenization complete.")
    print("Input IDs shape:", encodings['input_ids'].shape)
    print("Attention mask shape:", encodings['attention_mask'].shape)
    
    # Map emotion labels to integers using the defined mapping
    labels = torch.tensor(df['Emotion'].map(label_mapping).values)
    print("Labels tensor shape:", labels.shape)
    print("Unique label mapping:", label_mapping)
    
    # Create and return a TensorDataset
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    print(f"WASSA 2021 dataset loaded: {len(dataset)} samples")
    return dataset

def split_dataset(dataset, split_ratio=0.8, seed=42):
    """
    Split a TensorDataset into train and test subsets using a simple random split.
    """
    total_samples = len(dataset)
    train_size = int(split_ratio * total_samples)
    test_size = total_samples - train_size
    train_ds, test_ds = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"[Split] Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    return train_ds, test_ds

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

# --- Main Execution ---

if __name__ == "__main__":
    # Ensure the data folder exists
    os.makedirs("data", exist_ok=True)
    
    print("[Main] Loading and processing WASSA 2021 dataset...")
    wassa_tsv_path = "data/WASSA2021/wassa_2021.tsv"
    
    # Print the first 10 lines of the TSV file for debugging
    print("First 10 lines of the TSV file:")
    with open(wassa_tsv_path, "r", encoding="utf-8") as f:
        for i in range(10):
            print(f.readline().strip())
    
    # Load the dataset
    wassa_dataset = load_wassa(wassa_tsv_path, max_length=128)
    
    # Debug: print a sample from the dataset
    sample_idx = 0
    sample_input_ids, sample_attention, sample_label = wassa_dataset[sample_idx]
    print("\nSample from dataset:")
    print("Input IDs:", sample_input_ids.tolist())
    print("Attention mask:", sample_attention.tolist())
    print("Label:", sample_label.item())
    
    # Split dataset into train (80%) and test (20%) sets
    total_samples = len(wassa_dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    wassa_train, wassa_test = random_split(
        wassa_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    print("\nTrain/Test split:")
    print("Total samples:", total_samples)
    print("Train size:", len(wassa_train))
    print("Test size:", len(wassa_test))
    
    # Check label distribution in train and test splits
    def get_label_distribution(subset):
        labels_list = [wassa_dataset[idx][2].item() for idx in subset.indices]
        return pd.Series(labels_list).value_counts().sort_index()
    
    print("\nTrain label distribution:")
    print(get_label_distribution(wassa_train))
    print("\nTest label distribution:")
    print(get_label_distribution(wassa_test))
    
    # Save final datasets
    torch.save(wassa_train, 'data/preprocessed_dataset/wassa/train.pt')
    torch.save(wassa_test, 'data/preprocessed_dataset/wassa/test.pt')
    
    print("\nDatasets prepared and saved:")
    print(f"- WASSA 2021 dual-view train: {len(wassa_train)} samples")
    print(f"- WASSA 2021 test: {len(wassa_test)} samples")
