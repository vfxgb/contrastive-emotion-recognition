import torch
from torch.utils.data import DataLoader
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead
from contrastive_loss import SupConLoss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import random

# Configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 256
num_emotions = 9
batch_size = 256
num_epochs = 1000  # Reduced epochs for early stopping/validation
learning_rate = 2e-5

# Load original train dataset
from preprocess_data import load_crowdflower, split_dataset, DualViewDataset  # Ensure you import the new DualViewDataset

full_dataset, _, _, _ = load_crowdflower('data/CrowdFlower/text_emotion.csv')
train_ds, val_ds = split_dataset(full_dataset, split_ratio=0.8)

# Wrap training dataset with DualViewDataset for augmentation.
train_ds = DualViewDataset(train_ds, dropout_prob=0.1)

# Create DataLoader (for validation, you might use the original dataset without augmentation)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# For validation/evaluation, we use the first view only.
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Mamba config
mamba_args = dict(
    d_model=256,
    d_state=128,
    d_conv=4,
    expand=2,
)

# Initialize encoder & classifier
encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
classifier = ClassifierHead(embed_dim, num_emotions).to(device)

# Losses & optimizer
criterion_cls = CrossEntropyLoss()
criterion_contrastive = SupConLoss()
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)

# Training loop with dual-view contrastive learning

# Validation Function
def evaluate(encoder, classifier, dataloader, device):
    encoder.eval()
    classifier.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, _, labels in tqdm(dataloader, desc="Validation"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            embeddings = encoder(input_ids)
            logits = classifier(embeddings)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Training loop with validation
best_accuracy = 0
patience, trigger_times = 30, 0

for epoch in range(num_epochs):
    encoder.train()
    classifier.train()
    total_loss = 0
    
    for view1, view2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
        
        emb1, emb2 = encoder(view1), encoder(view2)
        features = torch.stack([emb1, emb2], dim=1)
        
        loss = criterion_cls(classifier(emb1), labels) + criterion_contrastive(features, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

    # Validation step
    val_accuracy = evaluate(encoder, classifier, val_loader, device)
    print(f"[Epoch {epoch+1}] Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping logic
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save({
            'encoder': encoder.state_dict(),
            'classifier': classifier.state_dict(),
        }, 'results/best_contrastive_mamba.pt')
        trigger_times = 0
        print(f"Best model saved at epoch {epoch+1} with accuracy: {val_accuracy:.4f}")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
