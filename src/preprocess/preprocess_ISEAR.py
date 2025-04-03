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
label_mapping = fetch_label_mapping(isear=True)

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_isear(csv_path, max_length=128):
    """
    Load and preprocess the ISEAR dataset from a CSV file.
    Assumes the CSV has columns 'Field1' and 'SIT'.
    'Field1' is used as the label and 'SIT' as the text.
    Only rows with labels present in label_mapping are kept.
    """
    print("Loading CSV from:", csv_path)
    # Load CSV using latin1 encoding and comma separator
    df = pd.read_csv(csv_path, encoding="latin1", sep=",")
    print("Original dataset shape:", df.shape)
    print("Columns in dataset:", df.columns.tolist())
    
    # Rename columns: use 'Field1' as 'Emotion' and 'SIT' as 'Text'
    df = df.rename(columns={'Field1': 'Emotion', 'SIT': 'Text'})
    print("Renamed columns: 'Field1' -> 'Emotion', 'SIT' -> 'Text'")
    
    # Show original label distribution based on 'Emotion'
    print("Original label distribution in 'Emotion':")
    print(df['Emotion'].value_counts())
    
    # Filter to keep only rows with desired emotions
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
    
    # Map emotion labels to integers
    labels = torch.tensor(df['Emotion'].map(label_mapping).values)
    print("Labels tensor shape:", labels.shape)
    print("Unique label mapping:", label_mapping)
    
    # Create and return a TensorDataset
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    print(f"ISEAR dataset loaded: {len(dataset)} samples")
    return dataset

def split_dataset(dataset, split_ratio=0.8, seed=42):
    """
    Split a TensorDataset into train and test subsets using stratified sampling.
    This function uses the input_ids as a proxy for uniqueness.
    """
    # Get tensors from dataset
    input_ids = dataset.tensors[0]
    labels = dataset.tensors[2]

    # Convert each row to a tuple (for immutability) to identify unique samples
    text_ids = [tuple(row.tolist()) for row in input_ids]
    _, indices = np.unique(text_ids, return_index=True, axis=0)

    # Stratified split using sklearn
    X_train_idx, X_test_idx = train_test_split(
        indices,
        train_size=split_ratio,
        random_state=seed,
        stratify=labels[indices].numpy()
    )

    from torch.utils.data import Subset
    train_ds = Subset(dataset, X_train_idx)
    test_ds = Subset(dataset, X_test_idx)

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
    os.makedirs("data", exist_ok=True)
    
    print("[Main] Loading and processing ISEAR dataset...")
    isear_csv_path = "data/ISEAR/isear_data.csv"
    
    # Print first 10 lines for debugging
    print("First 10 lines of the CSV file:")
    with open(isear_csv_path, "rb") as f:
        for i in range(10):
            print(f.readline())
    
    # Load the dataset
    isear_dataset = load_isear(isear_csv_path, max_length=128)
    
    # Debug: print a sample from the dataset
    sample_idx = 0
    sample_input_ids, sample_attention, sample_label = isear_dataset[sample_idx]
    print("\nSample from dataset:")
    print("Input IDs:", sample_input_ids.tolist())
    print("Attention mask:", sample_attention.tolist())
    print("Label:", sample_label.item())
    
    # Split dataset into train (80%) and test (20%) sets
    total_samples = len(isear_dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    isear_train, isear_test = random_split(
        isear_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    print("\nTrain/Test split:")
    print("Total samples:", total_samples)
    print("Train size:", len(isear_train))
    print("Test size:", len(isear_test))
    
    # Check for overlapping samples between train and test sets
    train_ids = set([tuple(isear_dataset[i][0].tolist()) for i in isear_train.indices])
    test_ids  = set([tuple(isear_dataset[i][0].tolist()) for i in isear_test.indices])
    overlap = train_ids.intersection(test_ids)
    print("Overlap between train and test indices (should be empty):", len(overlap))
    
    # Check label distributions in train and test splits
    def get_label_distribution(subset):
        labels_list = [isear_dataset[idx][2].item() for idx in subset.indices]
        return pd.Series(labels_list).value_counts().sort_index()
    
    print("\nTrain label distribution:")
    print(get_label_distribution(isear_train))
    print("\nTest label distribution:")
    print(get_label_distribution(isear_test))
    
    # Save final datasets
    torch.save(isear_train, 'data/preprocessed_dataset/isear/train.pt')
    torch.save(isear_test, 'data/preprocessed_dataset/isear/test.pt')
    
    print("\nDatasets prepared and saved:")
    print(f"- ISEAR train: {len(isear_train)} samples")
    print(f"- ISEAR test: {len(isear_test)} samples")
