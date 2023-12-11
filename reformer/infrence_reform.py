import torch
import torch.nn as nn
from torch.nn import functional as F
from reform import ReformerLanguageModel, encode, decode
import math

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
n_embd = 384             # Embedding dimension size
block_size = 256         # Maximum context length for predictions
n_layer = 6              # Number of Reformer layers
n_buckets = 32           # Number of buckets for LSH
n_hashes = 4  

model_save_path = 'reformer_language_model.pth'
model = ReformerLanguageModel(vocab_size, n_embd, block_size, n_layer, n_buckets, n_hashes)
model.load_state_dict(torch.load(model_save_path))
model.eval() 

stoi = { ch:i for i, ch in enumerate(chars) }

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def generate_text(initial_text,block_size=256, max_new_tokens=5000):
    idx = torch.tensor([encode(initial_text)], dtype=torch.long).to(device)
    generated_text = initial_text
    model.eval()
    
    with torch.no_grad():  
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]           
            probs = F.softmax(logits, dim=-1)  
            idx_next = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat((idx, idx_next), dim=1)
            last_token = decode(idx[0, -1].unsqueeze(0).tolist())
            generated_text += last_token
            print(last_token, end='', flush=True)
            
    return generated_text

initial_text = "Once upon a time" 
generate_text(initial_text)
