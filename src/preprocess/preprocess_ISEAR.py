import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from transformers import AutoTokenizer
import os
from utils import (
    clean_text,
    fetch_label_mapping,
    load_glove_embeddings,
    random_dropout_tokens,
    split_dataset,
    DualViewDataset,
)
from config import (
    BERT_MODEL,
    SPACY_MODEL,
    ISEAR_PATH,
    ISEAR_TEST_DS_PATH,
    ISEAR_TRAIN_DS_PATH,
    ISEAR_GLOVE_EMBEDDINGS_PATH
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with_glove = False
# fetch label mapping for ISEAR dataset
label_mapping = fetch_label_mapping(isear=True)

def load_isear_with_glove(csv_path, max_length=128):
    """
    Load and preprocess the ISEAR dataset from a CSV file.
    Assumes the CSV has columns 'Field1' and 'SIT'.
    'Field1' is used as the label and 'SIT' as the text.
    Only rows with labels present in label_mapping are kept.

    Args:
        csv_path (str): Path to the CSV file.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        TensorDataset: Dataset object containing input_ids, attention_masks, and labels.

    """
    # Initialize the BERT tokenizer
    tokenizer = Tokenizer(num_words=5000,oov_token="<UNK>")

    # Load CSV using latin1 encoding and comma separator
    df = pd.read_csv(csv_path, encoding="latin1", sep=",")
    print(f"[Isear] Loaded {len(df)} rows from {csv_path}")
    print("Original dataset shape:", df.shape)
    print("Columns in dataset:", df.columns.tolist())

    # Rename columns: use 'Field1' as 'Emotion' and 'SIT' as 'Text'
    df = df.rename(columns={"Field1": "Emotion", "SIT": "Text"})
    print("Renamed columns: 'Field1' -> 'Emotion', 'SIT' -> 'Text'")

    # Show original label distribution based on 'Emotion'
    print("Original label distribution in 'Emotion':")
    print(df["Emotion"].value_counts())

    # Filter to keep only rows with desired emotions
    df = df[df["Emotion"].isin(label_mapping.keys())].reset_index(drop=True)
    print("Filtered dataset shape:", df.shape)
    print("Filtered label distribution:")
    print(df["Emotion"].value_counts())

    # Clean the text in the 'Text' column
    df["content"] = df["Text"].apply(clean_text, extended=True)
    print("Sample cleaned text:", df["content"].iloc[0])

    texts = df["content"].tolist()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts) # Convert text to numerical sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    
    print("Tokenization complete.")

    # Map emotion labels to integers
    input_tensor = torch.tensor(padded_sequences)
    labels = torch.tensor(df["Emotion"].map(label_mapping).values)
    print("Labels tensor shape:", labels.shape)
    print("Unique label mapping:", label_mapping)

    # Create and return a TensorDataset
    dataset = TensorDataset(input_tensor, labels)
    print(f"ISEAR dataset loaded: {len(dataset)} samples")

    return dataset, tokenizer

def load_isear_without_glove(csv_path, max_length=128):
    """
    Load and preprocess the ISEAR dataset from a CSV file.
    Assumes the CSV has columns 'Field1' and 'SIT'.
    'Field1' is used as the label and 'SIT' as the text.
    Only rows with labels present in label_mapping are kept.

    Args:
        csv_path (str): Path to the CSV file.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        TensorDataset: Dataset object containing input_ids, attention_masks, and labels.

    """
    # Initialize the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    # Load CSV using latin1 encoding and comma separator
    df = pd.read_csv(csv_path, encoding="latin1", sep=",")
    print(f"[Isear] Loaded {len(df)} rows from {csv_path}")
    print("Original dataset shape:", df.shape)
    print("Columns in dataset:", df.columns.tolist())

    # Rename columns: use 'Field1' as 'Emotion' and 'SIT' as 'Text'
    df = df.rename(columns={"Field1": "Emotion", "SIT": "Text"})
    print("Renamed columns: 'Field1' -> 'Emotion', 'SIT' -> 'Text'")

    # Show original label distribution based on 'Emotion'
    print("Original label distribution in 'Emotion':")
    print(df["Emotion"].value_counts())

    # Filter to keep only rows with desired emotions
    df = df[df["Emotion"].isin(label_mapping.keys())].reset_index(drop=True)
    print("Filtered dataset shape:", df.shape)
    print("Filtered label distribution:")
    print(df["Emotion"].value_counts())

    # Clean the text in the 'Text' column
    df["content"] = df["Text"].apply(clean_text)
    print("Sample cleaned text:", df["content"].iloc[0])

    # Tokenize the cleaned text using the BERT tokenizer
    encodings = tokenizer(
        df["content"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    print("Tokenization complete.")
    print("Input IDs shape:", encodings["input_ids"].shape)
    print("Attention mask shape:", encodings["attention_mask"].shape)

    # Map emotion labels to integers
    labels = torch.tensor(df["Emotion"].map(label_mapping).values)
    print("Labels tensor shape:", labels.shape)
    print("Unique label mapping:", label_mapping)

    # Create and return a TensorDataset
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
    print(f"ISEAR dataset loaded: {len(dataset)} samples")

    return dataset


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("[Main] Loading and processing ISEAR dataset...")

    # Load the dataset
    if with_glove:
        isear_dataset, tokenizer = load_isear_with_glove(ISEAR_PATH, max_length=128)
        load_glove_embeddings(ISEAR_GLOVE_EMBEDDINGS_PATH, tokenizer)
    else:
        isear_dataset = load_isear_without_glove(ISEAR_PATH, max_length=128)

    # Split dataset into train (80%) and test (20%) sets
    total_samples = len(isear_dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    isear_train, isear_test = random_split(
        isear_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print("\nTrain/Test split:")
    print("Total samples:", total_samples)
    print("Train size:", len(isear_train))
    print("Test size:", len(isear_test))

    # Check label distributions in train and test splits
    def get_label_distribution(subset):
        labels_list = [isear_dataset[idx][2].item() for idx in subset.indices]
        return pd.Series(labels_list).value_counts().sort_index()

    print("\nTrain label distribution:")
    print(get_label_distribution(isear_train))
    print("\nTest label distribution:")
    print(get_label_distribution(isear_test))

    # Save final datasets
    torch.save(isear_train, ISEAR_TRAIN_DS_PATH)
    torch.save(isear_test, ISEAR_TEST_DS_PATH)

    print("\nDatasets prepared and saved:")
    print(f"- ISEAR train: {len(isear_train)} samples")
    print(f"- ISEAR test: {len(isear_test)} samples")
