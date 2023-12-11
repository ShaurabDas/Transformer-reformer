import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt import GPTLanguageModel, encode, decode
import streamlit as st


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))

model_save_path = 'gpt_language_model.pth'
model = GPTLanguageModel()
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Set the model to evaluation mode

stoi = { ch:i for i, ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s]

# Device configuration (CPU or CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

st.title("GPT Shakespeare Text Generator")
user_input = st.text_area("Enter the starting text:", "Once upon a time")


# Define the inference function
def generate_text(initial_text,block_size=256, max_new_tokens=5000):
    # Encode the initial text
    idx = torch.tensor([encode(initial_text)], dtype=torch.long).to(device)

    # Initialize the generated text with the initial text
    generated_text = initial_text

    text_placeholder = st.empty()
    text_placeholder.text(generated_text)

    # Generate text token by token
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # Disable gradient calculations for inference
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # Get the predictions
            logits, _ = model(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            # Decode the last token and add it to the generated text
            last_token = decode(idx[0, -1].unsqueeze(0).tolist())
            generated_text += last_token

            # Print the newly generated token
            #print(last_token, end='', flush=True)
            text_placeholder.text(generated_text)

            if last_token == '<EOS>':  # End the loop if the end of sequence token is generated
                break
    return generated_text

if st.button('Generate Text'):
    generate_text(user_input, max_new_tokens=5000) 
