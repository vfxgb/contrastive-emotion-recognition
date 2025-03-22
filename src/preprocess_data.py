import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_and_preprocess(path, max_length=128):
    df = pd.read_csv(path)
    texts = df['content'].tolist()
    labels = df['sentiment'].tolist()
    
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
    
    dataset = TensorDataset(
        torch.tensor(encodings['input_ids']),
        torch.tensor(encodings['attention_mask']),
        torch.tensor(labels)
    )
    return dataset

if __name__ == "__main__":
    train_dataset = load_and_preprocess('data/CrowdFlower/text_emotion.csv')
    test_dataset = load_and_preprocess('data/WASSA2017/test.csv')

    torch.save(train_dataset, 'data/train.pt')
    torch.save(test_dataset, 'data/test.pt')
