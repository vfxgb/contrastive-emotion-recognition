import torch
from torch.utils.data import DataLoader
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead
from contrastive_loss import SupConLoss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Subset
torch.serialization.add_safe_globals([Subset])

# Configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 256
num_emotions = 13
batch_size = 32
num_epochs = 1000
learning_rate = 2e-5

# Load dataset
train_dataset = torch.load('data/train.pt', weights_only=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Mamba config (Mamba from pip uses direct kwargs)
mamba_args = dict(
    d_model=128,   # input embedding dim
    d_state=64,    # state space dim
    d_conv=4,      # convolution width
    expand=2       # block expansion factor
)

# Initialize encoder & classifier
encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
classifier = ClassifierHead(embed_dim, num_emotions).to(device)

# Losses & optimizer
criterion_cls = CrossEntropyLoss()
criterion_contrastive = SupConLoss()
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)

# Training loop
encoder.train()
classifier.train()
for epoch in range(num_epochs):
    total_loss = 0
    for input_ids, _, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids, labels = input_ids.to(device), labels.to(device)

        # Forward pass
        emotion_emb = encoder(input_ids)
        logits = classifier(emotion_emb)

        # Loss computation
        loss_cls = criterion_cls(logits, labels)
        loss_contrastive = criterion_contrastive(emotion_emb, labels)
        loss = loss_cls + loss_contrastive

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

# Save model components
torch.save({
    'encoder': encoder.state_dict(),
    'classifier': classifier.state_dict(),
}, 'results/contrastive_mamba_decoupled.pt')