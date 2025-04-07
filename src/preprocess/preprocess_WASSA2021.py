import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
import os
from utils import clean_text, fetch_label_mapping, split_dataset, load_glove_embeddings
from config import BERT_MODEL, WASSA_PATH, WASSA_TEST_DS_PATH_WITHOUT_GLOVE, WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE, WASSA_GLOVE_EMBEDDINGS_PATH, WASSA_TEST_DS_PATH_WITH_GLOVE, WASSA_TRAIN_DS_PATH_WITH_GLOVE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse

# get label mapping for WASSA dataset
label_mapping = fetch_label_mapping(wassa=True)

def load_wassa_with_glove(tsv_path, max_length=128):
    """
    Load and preprocess the WASSA 2021 dataset from a TSV file.
    Renames 'emotion_labels' to 'Emotion' and 'essays' to 'Text'.
    Filters to keep only rows with one of the six Ekman emotions,
    cleans the text, tokenizes it, and maps the labels.
    Returns a TensorDataset.
    """
    tokenizer = Tokenizer(num_words=5000,oov_token="<UNK>")

    print("Loading TSV from:", tsv_path)
    df = pd.read_csv(tsv_path, sep="\t")
    print("Original dataset shape:", df.shape)
    print("Columns in dataset:", df.columns.tolist())

    # Rename columns for consistency
    df = df.rename(columns={"emotion_label": "Emotion", "essay": "Text"})

    # Filter to keep only rows with desired emotions (exclude others such as 'neutral')
    df = df[df["Emotion"].isin(label_mapping.keys())].reset_index(drop=True)
    print("Filtered dataset shape:", df.shape)
    print("Filtered label distribution:")
    print(df["Emotion"].value_counts())

    # Clean the text in the 'Text' column
    df["content"] = df["Text"].apply(clean_text, extended = True)
    texts = df["content"].tolist()

    # Tokenize the cleaned text using the BERT tokenizer
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts) # Convert text to numerical sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    input_tensor = torch.tensor(padded_sequences)

    # Map emotion labels to integers using the defined mapping
    labels = torch.tensor(df["Emotion"].map(label_mapping).values)

    # Create and return a TensorDataset
    dataset = TensorDataset(input_tensor, labels)
    print(f"WASSA 2021 dataset loaded: {len(dataset)} samples")

    return dataset, tokenizer

def load_wassa_without_glove(tsv_path, max_length=128):
    """
    Load and preprocess the WASSA 2021 dataset from a TSV file.
    Renames 'emotion_labels' to 'Emotion' and 'essays' to 'Text'.
    Filters to keep only rows with one of the six Ekman emotions,
    cleans the text, tokenizes it, and maps the labels.
    Returns a TensorDataset.
    """
    # Initialize the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    print("Loading TSV from:", tsv_path)
    df = pd.read_csv(tsv_path, sep="\t")
    print("Original dataset shape:", df.shape)
    print("Columns in dataset:", df.columns.tolist())

    # Rename columns for consistency
    df = df.rename(columns={"emotion_label": "Emotion", "essay": "Text"})

    # Filter to keep only rows with desired emotions (exclude others such as 'neutral')
    df = df[df["Emotion"].isin(label_mapping.keys())].reset_index(drop=True)

    # Clean the text in the 'Text' column
    df["content"] = df["Text"].apply(clean_text)

    # Tokenize the cleaned text using the BERT tokenizer
    encodings = tokenizer(
        df["content"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # Map emotion labels to integers using the defined mapping
    labels = torch.tensor(df["Emotion"].map(label_mapping).values)
    print("Labels tensor shape:", labels.shape)
    print("Unique label mapping:", label_mapping)

    # Create and return a TensorDataset
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
    print(f"WASSA 2021 dataset loaded: {len(dataset)} samples")

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_glove", action="store_true", help="Use GloVe embeddings")
    args = parser.parse_args()
    with_glove = args.with_glove
    
    # Ensure the data folder exists
    os.makedirs("data", exist_ok=True)

    print("[Main] Loading and processing WASSA 2021 dataset...")

    if with_glove:
        wassa_dataset, tokenizer = load_wassa_with_glove(WASSA_PATH, max_length=128)
        load_glove_embeddings(tokenizer, WASSA_GLOVE_EMBEDDINGS_PATH)

        print("[Main] Splitting dataset into train and test...")
        train_ds, test_ds = split_dataset(wassa_dataset, split_ratio=0.8, glove=True)

        print("[Main] Saving datasets to disk...")
        torch.save(train_ds, WASSA_TRAIN_DS_PATH_WITH_GLOVE )
        torch.save(test_ds, WASSA_TEST_DS_PATH_WITH_GLOVE)
        
    else:
        wassa_dataset = load_wassa_without_glove(WASSA_PATH, max_length=128)

        print("[Main] Splitting dataset into train and test...")
        train_ds, test_ds = split_dataset(wassa_dataset, split_ratio=0.8, glove=False)

        print("[Main] Saving datasets to disk...")
        torch.save(train_ds, WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE)
        torch.save(test_ds, WASSA_TEST_DS_PATH_WITHOUT_GLOVE)

    print("[Main] Done.")