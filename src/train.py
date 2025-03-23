# import torch
# from torch.utils.data import DataLoader
# from models.contrastive_model import ContrastiveMambaModel
# from contrastive_loss import SupConLoss
# from torch.nn import CrossEntropyLoss
# from tqdm import tqdm
# from torch.utils.data import TensorDataset
# from models.mamba import ModelArgs
# from transformers import AutoTokenizer

# torch.serialization.add_safe_globals([TensorDataset])
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# train_dataset = torch.load('data/train.pt')
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# mamba_args = ModelArgs(d_model=128, n_layer=2, vocab_size=30522)
# model = ContrastiveMambaModel(mamba_args, num_emotions=13).to(device)
# # from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# # num_labels = 13  # the number of labels

# # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
# # model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b")

# # model.lm_head = torch.nn.Linear(model.config.d_model, num_labels)
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# criterion_cls = CrossEntropyLoss()
# criterion_contrastive = SupConLoss()

# model.train()
# for epoch in range(100000):
#     epoch_loss = 0
#     for input_ids, _, labels in tqdm(train_loader):  # Ignore attention_mask
#         input_ids = input_ids.to(device)
#         labels = labels.to(device)

#         logits, embeddings = model(input_ids)

#         loss_cls = criterion_cls(logits, labels)
#         loss_contrastive = criterion_contrastive(embeddings, labels)
#         loss = loss_cls + loss_contrastive

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#     print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")

# torch.save(model.state_dict(), 'results/contrastive_mamba.pt')

import torch
from torch.utils.data import DataLoader
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead
from contrastive_loss import SupConLoss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.data import TensorDataset
from models.mamba import ModelArgs
from transformers import AutoTokenizer

torch.serialization.add_safe_globals([TensorDataset])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = torch.load('data/train.pt')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

mamba_args = ModelArgs(d_model=128, n_layer=2, vocab_size=30522)
embed_dim = 256
num_emotions = 13  
num_epochs = 100000

# Instantiate encoder and classifier separately.
encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
classifier = ClassifierHead(embed_dim, num_emotions).to(device)

params = list(encoder.parameters()) + list(classifier.parameters())
optimizer = torch.optim.AdamW(params, lr=2e-5)

criterion_cls = CrossEntropyLoss()
criterion_contrastive = SupConLoss()

encoder.train()
classifier.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for input_ids, _, labels in tqdm(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Get embeddings from encoder.
        emotion_emb = encoder(input_ids)
        # Get logits from classifier head.
        logits = classifier(emotion_emb)
        
        loss_cls = criterion_cls(logits, labels)
        loss_contrastive = criterion_contrastive(emotion_emb, labels)
        loss = loss_cls + loss_contrastive

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")

# Save the model components separately.
torch.save({
    'encoder': encoder.state_dict(),
    'classifier': classifier.state_dict(),
}, 'results/contrastive_mamba_decoupled.pt')
