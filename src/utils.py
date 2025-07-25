import torch
import numpy as np
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch import nn
import random
import re
import string
import spacy
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from config import GLOVE_PATH, SEED
import os

nlp = spacy.load("en_core_web_sm")


def get_versioned_path(base_path, finetune_mode):
    """
    Adds finetune_mode tag to filename before extension and returns the new path.

    Args:
        base_path (str): the original file name
        finetune_mode (int): the number to append

    Returns:
        str: new file path name
    """
    base_dir, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{finetune_mode}{ext}"

    return os.path.join(base_dir, new_filename)


def random_dropout_tokens(token_ids, dropout_prob=0.1):
    """
    Simple augmentation.
    Randomly drop tokens from a sequence with a given probability (except special tokens).
    Assumes special tokens: [CLS]=101, [SEP]=102, [PAD]=0.

    Args:
        token_ids (list): the token IDs of the input sequence.
        dropout_prob (float): probability of dropping a token.

    Returns:
        list: list of token IDs after applying dropout.
    """
    return [
        tok
        for tok in token_ids
        if random.random() > dropout_prob or tok in [101, 102, 0]
    ]


def print_test_stats(test_acc_list, test_f1_list, num_runs):
    """
    Computes and prints the mean and standard deviation of evaluation metrics across multiple runs.

    Args:
        test_acc_list (list): List of test accuracy values for each run.
        test_f1_list (list): List of test F1 score values for each run.
        num_runs (int): Number of evaluation runs.
    """
    mean_test_acc, std_test_acc = np.mean(test_acc_list), np.std(test_acc_list)
    mean_test_f1, std_test_f1 = np.mean(test_f1_list), np.std(test_f1_list)

    print(f"\nFinal Test Accuracy over {num_runs} runs: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"Final Test F1 Score over {num_runs} runs: {mean_test_f1:.4f} ± {std_test_f1:.4f}")

def split_dataset(dataset, split_ratio=0.8, seed=SEED, glove=True):
    """
    Splits a TensorDataset (or its Subset) into training and testing subsets.

    For GloVe-based datasets, it assumes a 2-tensor structure (input_ids, labels).
    For BERT-based datasets, it assumes a 3-tensor structure (input_ids, attention_masks, labels).

    Duplicate input sequences are removed before stratified splitting.

    Args:
        dataset (TensorDataset or Subset): The dataset to be split.
        split_ratio (float): Proportion of the data to include in the train split (default: 0.8).
        seed (int): Random seed for reproducibility (default: 42).
        glove (bool): Whether the dataset format is GloVe-based (True) or BERT-based (False).

    Returns:
        Tuple[Subset, Subset]: A tuple containing the train and test subsets.

    """
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        subset_indices = dataset.indices
    else:
        base_dataset = dataset
        subset_indices = list(range(len(dataset)))

    input_ids = base_dataset.tensors[0][subset_indices]
    if glove:
        labels = base_dataset.tensors[1][subset_indices]
    else:
        attention_mask = base_dataset.tensors[1][subset_indices]
        labels = base_dataset.tensors[2][subset_indices]

    text_ids = [tuple(row.tolist()) for row in input_ids]
    _, indices = np.unique(text_ids, return_index=True, axis=0)

    selected_indices = [subset_indices[i] for i in indices]

    X_train_idx, X_test_idx = train_test_split(
        selected_indices,
        train_size=split_ratio,
        random_state=seed,
        stratify=labels[indices].numpy(),
    )

    train_ds = Subset(base_dataset, X_train_idx)
    test_ds = Subset(base_dataset, X_test_idx)

    return train_ds, test_ds


class DualViewDataset(torch.utils.data.Dataset):
    """
    A dataset class used for contrastive learning, where each sample consists of two augmented views.

    Args:
        subset (torch.utils.data.Dataset): The dataset to augment.
        dropout_prob (float): Probability of dropping a token during augmentation.
    """

    def __init__(self, subset, dropout_prob=0.1):
        # Handles both TensorDataset and Subset objects

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


def set_seed(seed):
    """
    Sets random seed.

    Args:
        seed(int) : random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_glove_embeddings(tokenizer, embeddings_file_path):
    """
    Loads pre-trained GloVe word embeddings and constructs an embedding matrix
    aligned with the tokeniser's vocabulary

    For each word in the tokeniser vocabulary, the corresponding GloVe embedding
    is used. If the word is not found in gloVe, a random vector is assigned.

    Args:
        tokenizer (Tokenizer): Fitted Keras tokenizer with word_index.
        embeddings_file_path (str): Path to save the geneerate .npy file containing the embedding matrix and vocab size.

    Saves:
        A NumPy `.npy` file with keys:
            - "embedding_matrix": A matrix of shape (vocab_size, 300)
            - "vocab_size": The size of the vocabulary
    """
    print("Loading GloVe embeddings...")
    embeddings_index = {}

    with open(GLOVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = (
                vector  # reads the GLOVE file and gets all the embedding for each word
            )

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, 300))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        embedding_matrix[i] = (
            embedding_vector if embedding_vector is not None else np.random.randn(300)
        )

    vocab_size = len(tokenizer.word_index) + 1
    np.save(
        embeddings_file_path,
        {"embedding_matrix": embedding_matrix, "vocab_size": vocab_size},
    )


def fetch_label_mapping(isear=False, crowdflower=False, wassa=False):
    """
    Fetches the label mapping for the specified dataset.
    Args:
        isear : if True, return label mapping for ISEAR dataset.
        crowdflower : if True, return label mapping for CrowdFlower dataset.
        wassa : if True, return label mapping for WASSA dataset.

    """
    if isear:
        # return label mapping for isear dataset
        return {
            "anger": 0,
            "sadness": 1,
            "disgust": 2,
            "shame": 3,
            "fear": 4,
            "joy": 5,
            "guilt": 6,
        }
    elif wassa:
        # return label mapping for wassa dataset
        return {
            "anger": 0,
            "sadness": 1,
            "disgust": 2,
            "fear": 3,
            "joy": 4,
            "surprise": 5,
        }


def clean_text(text, extended=False):
    """
    Clean text by removing URLs, mentions, hashtags, extra whitespace,
    and converting to lowercase.
    Args:
        text : the text that is to be cleaned.
        extended : if True, applies lematisation and removes
            punctuations
    """
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#", "", text)  # Remove hashtag symbols
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    if not extended:
        return text.lower()

    doc = nlp(text.lower())

    tokens = [token.lemma_ for token in doc if token.text not in string.punctuation]
    return " ".join(tokens)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    This loss function is used for contrastive learning tasks where the model
    learns to embed similar samples closer together and dissimilar samples further apart
    by comparing their embeddings.

    """

    def __init__(self, temperature=0.07, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, labels):
        device = features.device

        if features.dim() == 3:
            B, views, D = features.shape
            features = features.view(B * views, D)
            labels = labels.repeat(views)

        features = F.normalize(features, dim=1)

        sim = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0)

        exp_sim = torch.exp(sim)
        numerator = (exp_sim * mask).sum(dim=1)
        denominator = exp_sim.sum(dim=1)

        pos_count = mask.sum(dim=1)
        valid = pos_count > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -torch.log(
            (numerator[valid] + self.eps) / (denominator[valid] + self.eps)
        )
        return loss.mean()
