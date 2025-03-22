import torch
from torch.utils.data import DataLoader
from contrastive_model import ContrastiveMambaModel
from contrastive_loss import SupConLoss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = torch.load('data/train.pt')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

mamba_args = ModelArgs(d_model=128, n_layer=2, vocab_size=30522)
model = ContrastiveMambaModel(mamba_args, num_emotions=6).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion_cls = CrossEntropyLoss()
criterion_contrastive = SupConLoss()

model.train()
for epoch in range(5):
    epoch_loss = 0
    for input_ids, attn_mask, labels in tqdm(train_loader):
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits, embeddings = model(input_ids)

        loss_cls = criterion_cls(logits, labels)
        loss_contrastive = criterion_contrastive(embeddings, labels)

        loss = loss_cls + loss_contrastive

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")

torch.save(model.state_dict(), 'results/contrastive_mamba.pt')
