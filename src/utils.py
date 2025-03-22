import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_emb = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title('t-SNE visualization of emotion embeddings')
    plt.show()
