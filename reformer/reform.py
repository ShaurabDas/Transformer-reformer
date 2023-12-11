import random
from torch.nn import functional as F
import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)} 
itos = {i: ch for i, ch in enumerate(chars)}  


encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  
train_data, val_data = data[:n], data[n:]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(data, block_size, batch_size):
    # Function to generate a batch of data
    length = data.size(0) - block_size
    if length <= 0:
        raise ValueError("Block size is too large.")

    start_indices = torch.randint(length, size=(batch_size,))
    sequences = [data[i:i + block_size] for i in start_indices]
    x = torch.stack(sequences).to(device)  # Input sequence
    y = torch.stack([data[i + 1:i + block_size + 1] for i in start_indices]).to(device)  # Target sequence
    return x, y


class LSHAttention(nn.Module):
    def __init__(self, n_embd, n_buckets, n_hashes, causal=False):
        super().__init__()
        self.n_embd = n_embd
        self.n_buckets = n_buckets
        self.n_hashes = n_hashes
        self.causal = causal
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        
        self.projections = nn.Parameter(torch.randn(self.n_hashes, self.n_embd))
        
     
    def hash_vectors(self, vecs):
        # Compute hash: project vectors and take sign
        # Multiply by projections, resulting in (n_hashes, B, T, C)
        vecs = vecs.to(torch.float32)
        projections = torch.einsum('he,bte->hbt', self.projections, vecs)
        # Sign of the projections (1 or -1)
        hash_values = projections.sign()

        # Restrict bucket numbers to be within the range of n_buckets
        buckets = torch.einsum('hbt,h->bt', (hash_values + 1) / 2, 2**torch.arange(self.n_hashes, device=vecs.device).float())
        buckets = buckets.long() % self.n_buckets
        return buckets

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q_buckets = self.hash_vectors(q)
        k_buckets = self.hash_vectors(k)
        # For simplicity, using regular dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.n_embd)
        attn = F.softmax(attn_weights, dim=-1)
        v = self.value(x)
        out = torch.matmul(attn, v)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.activation = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ReversibleLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, rev=False):
        x1, x2 = x.chunk(2, dim=-1)
        if not rev:
            y1 = x1 + self.fn(x2)
            y2 = x2 + self.fn(y1)
        else:
            y2 = x2 - self.fn(x1)
            y1 = x1 - self.fn(y2)
        return torch.cat([y1, y2], dim=-1)

class ReformerBlock(nn.Module):
    def __init__(self, n_embd, n_buckets, n_hashes, causal=False):
        super().__init__()
        self.attn = LSHAttention(n_embd, n_buckets, n_hashes, causal)
        self.ffwd = FeedForward(n_embd)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.norm1(x)
        y = self.attn(y) + x 
        y = self.norm2(y)
        y = self.ffwd(y) + y 
        return y
    
class ReformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_buckets, n_hashes, causal=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.layers = nn.ModuleList([ReformerBlock(n_embd, n_buckets, n_hashes, causal) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_embedding(idx) + self.positional_embedding[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
if __name__ == "__main__":
    n_embd = 384
    n_head = 6
    n_layer = 6
    n_buckets = 32
    n_hashes = 4
    block_size = 256
    batch_size = 64
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model instantiation
    model = ReformerLanguageModel(vocab_size, n_embd, block_size, n_layer, n_buckets, n_hashes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

#     Function to estimate loss (average loss over several batches)
    @torch.no_grad()
    def estimate_loss(split):
        total_loss = 0.0
        total_batches = 0
        data = train_data if split == 'train' else val_data

        for i in range(0, len(data) - block_size, batch_size):
            x, y = get_batch(data, block_size, batch_size)
            logits, loss = model(x, y)
            total_loss += loss.item()
            total_batches += 1

        return total_loss / total_batches

    train_losses = []
    val_losses = []
    for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss('val')
#                 print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                print(f"step {iter}, Loss {losses:.4f}")


            xb, yb = get_batch(train_data, block_size, batch_size)

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


    torch.save(model.state_dict(), 'reformer_language_model.pth')

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


