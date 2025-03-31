import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead
from contrastive_loss import SupConLoss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 256
num_emotions = 7
batch_size = 256
num_epochs = 1000
learning_rate = 1e-3
num_runs = 5

# Seed function for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    return correct / total

# Run multiple trials
all_test_accuracies = []

for run in range(num_runs):
    print(f"\n🔁 Run {run+1}/{num_runs}")
    set_seed(42 + run)

    # Load dataset
    train_ds = torch.load('data/train.pt', weights_only=False)
    test_ds = torch.load('data/test.pt', weights_only=False)

    # Split train into train/val
    train_len = int(0.9 * len(train_ds))
    val_len = len(train_ds) - train_len
    train_ds, val_ds = random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42 + run))

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model initialization
    mamba_args = dict(d_model=256, d_state=128, d_conv=4, expand=2)
    encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
    classifier = ClassifierHead(embed_dim, num_emotions).to(device)

    # Load pretrained encoder
    checkpoint = torch.load('results/fully_trained/contrastive_mamba_decoupled.pt')
    encoder.load_state_dict(checkpoint['encoder'])

    # Loss and optimizer
    criterion_cls = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)

    # Fine-tuning
    best_val_acc = 0
    for epoch in range(num_epochs):
        encoder.train()
        classifier.train()
        total_loss = 0

        for input_ids, _, labels in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            embeddings = encoder(input_ids)
            logits = classifier(embeddings)
            loss = criterion_cls(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(encoder, classifier, val_loader, device)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_encoder = encoder.state_dict()
            best_classifier = classifier.state_dict()

    # Load best model and evaluate on test set
    encoder.load_state_dict(best_encoder)
    classifier.load_state_dict(best_classifier)
    test_acc = evaluate(encoder, classifier, test_loader, device)
    all_test_accuracies.append(test_acc)
    print(f"✅ Run {run+1} Test Accuracy: {test_acc:.4f}")

# Print final results
mean_acc = np.mean(all_test_accuracies)
std_acc = np.std(all_test_accuracies)
print(f"\n📊 Final Test Accuracy over {num_runs} runs: {mean_acc:.4f} ± {std_acc:.4f}")

