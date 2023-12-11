import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt import GPTLanguageModel, encode, decode


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

model_save_path = 'gpt_language_model.pth'
model = GPTLanguageModel()
model.load_state_dict(torch.load(model_save_path))
model.eval() 

stoi = { ch:i for i, ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s]


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

# Example usage
initial_text = "Once upon a time" 
generate_text(initial_text)
