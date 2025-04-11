import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
import os
from utils import clean_text, fetch_label_mapping, split_dataset, load_glove_embeddings
from config import (
    BERT_MODEL,
    WASSA_PATH,
    WASSA_TEST_DS_PATH_WITHOUT_GLOVE,
    WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE,
    WASSA_GLOVE_EMBEDDINGS_PATH,
    WASSA_TEST_DS_PATH_WITH_GLOVE,
    WASSA_TRAIN_DS_PATH_WITH_GLOVE,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse

# get label mapping for WASSA dataset
label_mapping = fetch_label_mapping(wassa=True)

def load_wassa_with_glove(path, max_length=128):
    """
    Load and preprocess the WASSA 2021 dataset from a TSV file for use with GloVe embeddings.

    Renames 'emotion_labels' to 'Emotion' and 'essays' to 'Text'.
    Removes neutral emotions.
    Filters to keep only rows with one of the six Ekman emotions,
    cleans the text, tokenizes it, and maps the labels.
    
    Args:
        path (str): Path to the TSV file.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        TensorDataset: Dataset object containing input_ids and labels.
        Tokenizer: Fitted Keras Tokeniser object

    """
    # initialise tokenizer
    tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")

    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"emotion_label": "Emotion", "essay": "Text"})

    # keep only rows with desired emotions 
    df = df[df["Emotion"].isin(label_mapping.keys())].reset_index(drop=True)

    # Clean text
    df["content"] = df["Text"].apply(clean_text, extended=True)
    texts = df["content"].tolist()

    # Tokenize the cleaned text using the BERT tokenizer
    # Convert text to numerical sequences
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(
        sequences, maxlen=max_length, padding="post", truncating="post"
    )
    input_tensor = torch.tensor(padded_sequences)

    # Map emotion labels to integers using the defined mapping
    labels = torch.tensor(df["Emotion"].map(label_mapping).values)

    # Create and return a TensorDataset
    dataset = TensorDataset(input_tensor, labels)
    print(
        f"[WASSA] Loaded crowdflower dataset: {len(dataset)} samples from {path}"
    )
    print(f"[WASSA] Label map: {label_mapping}")

    return dataset, tokenizer


def load_wassa_without_glove(path, max_length=128):
    """
    Load and preprocess the WASSA 2021 dataset from a TSV file.

    Renames 'emotion_labels' to 'Emotion' and 'essays' to 'Text'.
    Removes neutral emotions.
    Filters to keep only rows with one of the six Ekman emotions,
    cleans the text, tokenizes it, and maps the labels.
    
    Args:
        path (str): Path to the TSV file.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        TensorDataset: Dataset object containing input_ids, attention_masks, and labels.
    """

    # Initialize the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"emotion_label": "Emotion", "essay": "Text"})

    # keep only rows with desired emotions 
    df = df[df["Emotion"].isin(label_mapping.keys())].reset_index(drop=True)

    # Clean the text 
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

    # Create and return a TensorDataset
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
    print(
        f"[WASSA] Loaded crowdflower dataset: {len(dataset)} samples from {path}"
    )
    print(f"[WASSA] Label map: {label_mapping}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # when --with_glove is set, the dataset is preprocessed for use with GloVe embeddings.
    parser.add_argument(
        "--with_glove", action="store_true", help="Use GloVe embeddings"
    )

    # when --force_preprocess is set, the dataset is processed again even when the train.pt and test.pt are present
    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Force reprocessing even if files exist",
    )

    args = parser.parse_args()
    with_glove = args.with_glove
    force_preprocess = args.force_preprocess

    # Ensure the data folder exists
    os.makedirs("data/preprocessed_dataset/wassa/", exist_ok=True)

    print("[Main] Loading and processing WASSA 2021 dataset...")

    if with_glove:
        if (
            not force_preprocess
            and os.path.exists(WASSA_TRAIN_DS_PATH_WITH_GLOVE)
            and os.path.exists(WASSA_TEST_DS_PATH_WITH_GLOVE)
        ):
            print("[Main] Dataset already preprocessed. Skipping...")
        else:
            print("[Main] Preprocessing Dataset.")
            wassa_dataset, tokenizer = load_wassa_with_glove(WASSA_PATH, max_length=128)
            load_glove_embeddings(tokenizer, WASSA_GLOVE_EMBEDDINGS_PATH)

            print("[Main] Splitting dataset into train and test...")
            train_ds, test_ds = split_dataset(
                wassa_dataset, split_ratio=0.8, glove=True
            )

            print("[Main] Saving datasets to disk...")
            torch.save(train_ds, WASSA_TRAIN_DS_PATH_WITH_GLOVE)
            torch.save(test_ds, WASSA_TEST_DS_PATH_WITH_GLOVE)

    else:
        if (
            not force_preprocess
            and os.path.exists(WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE)
            and os.path.exists(WASSA_TEST_DS_PATH_WITHOUT_GLOVE)
        ):
            print("[Main] Dataset already preprocessed. Skipping...")
        else:
            print("[Main] Preprocessing Dataset.")
            wassa_dataset = load_wassa_without_glove(WASSA_PATH, max_length=128)

            print("[Main] Splitting dataset into train and test...")
            train_ds, test_ds = split_dataset(
                wassa_dataset, split_ratio=0.8, glove=False
            )

            print("[Main] Saving datasets to disk...")
            torch.save(train_ds, WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE)
            torch.save(test_ds, WASSA_TEST_DS_PATH_WITHOUT_GLOVE)

    print("[Main] Done.")
