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

# import matplotlib.pyplot as plt
# def visualize_embeddings(embeddings, labels):
#     tsne = TSNE(n_components=2, random_state=42)
#     reduced_emb = tsne.fit_transform(embeddings)

#     plt.figure(figsize=(8, 8))
#     scatter = plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels, cmap='tab10', alpha=0.7)
#     plt.legend(*scatter.legend_elements(), title="Classes")
#     plt.title('t-SNE visualization of emotion embeddings')
#     plt.show()

nlp = spacy.load("en_core_web_sm")


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


def split_dataset(dataset, split_ratio=0.8, seed=42):
    """
    Splits the dataset into training and test sets while maintaining label distribution using stratified sampling.

    Args:
        dataset (TensorDataset): The dataset to split.
        split_ratio (float): The ratio of the data to be used for training.
        seed (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training and test datasets as Subset objects.
    """
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
        stratify=labels[
            indices
        ].numpy(),  # Stratified sampling to maintain label distribution
    )

    train_ds = Subset(dataset, X_train_idx)
    test_ds = Subset(dataset, X_test_idx)

    print(f"[Split] Train size: {len(train_ds)}, Test size: {len(test_ds)}")
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


def clean_text(text, extended=True):
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
