import torch
from torch.utils.data import DataLoader
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead
from contrastive_loss import SupConLoss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 256
num_emotions = 4
batch_size = 256
num_epochs = 1000
learning_rate = 2e-5

# Load prepared datasets
train_ds = torch.load('data/train.pt', weights_only=False)
val_ds = torch.load('data/val.pt', weights_only=False)
test_ds = torch.load('data/test.pt', weights_only=False)

# Data augmentation wrapper
from preprocess_generalization import DualViewDataset
train_ds_augmented = DualViewDataset(train_ds, dropout_prob=0.1)

# Dataloaders
train_loader = DataLoader(train_ds_augmented, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Load trained encoder checkpoint
checkpoint = torch.load('results/fully_trained/contrastive_mamba_decoupled.pt') # remove if not using the weights

# Mamba Configuration
mamba_args = dict(
    d_model=256,
    d_state=128,
    d_conv=4,
    expand=2,
)

# Model initialization
encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)

encoder.load_state_dict(checkpoint['encoder'])
# We freeze our previously trained encoder weights
for param in encoder.parameters():
    param.requires_grad = False  # Set to True if fine-tuning desired

classifier = ClassifierHead(embed_dim, num_emotions).to(device)

# Loss and Optimizer
criterion_cls = CrossEntropyLoss()
criterion_contrastive = SupConLoss()
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)

# Evaluation function
def evaluate(encoder, classifier, dataloader, device):
    encoder.eval()
    classifier.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for input_ids, attention_masks, labels in tqdm(dataloader, desc="Evaluating"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            embeddings = encoder(input_ids)
            logits = classifier(embeddings)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

# Training Loop with Early Stopping
best_val_accuracy = 0
patience, trigger_times = 30, 0

for epoch in range(num_epochs):
    encoder.train()
    classifier.train()

    total_loss = 0
    for view1, view2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

        emb1, emb2 = encoder(view1), encoder(view2)
        features = torch.stack([emb1, emb2], dim=1)

        loss_contrastive = criterion_contrastive(features, labels)
        loss_cls = criterion_cls(classifier(emb1), labels)

        loss = loss_cls + loss_contrastive

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

    val_accuracy = evaluate(encoder, classifier, val_loader, device)
    print(f"[Epoch {epoch+1}] Validation Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save({
            'encoder': encoder.state_dict(),
            'classifier': classifier.state_dict(),
        }, 'results/best_generalization_mamba.pt')
        trigger_times = 0
        print(f"[Epoch {epoch+1}] New best model saved.")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# Final generalization evaluation on WASSA
test_accuracy = evaluate(encoder, classifier, test_loader, device)
print(f"Final Generalization Accuracy (WASSA): {test_accuracy:.4f}")
