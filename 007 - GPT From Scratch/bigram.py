import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
train_test_split = 0.9
batch_size = 32
block_size = 8
max_iters  = 3000
eval_interval = 300
learning_rate = 1e-2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


torch.manual_seed(1337)

# Load Tiny Shakespeare dataset 
# (also refer to Andrej Karpathy's blog: http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
with open('tiny-shakespeare.txt', 'r') as f:
    text = f.read()

# Find all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings from characters to indices and vice versa
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[ch] for ch in s]        # Take a string, return a list of indices/integers
decode = lambda l: ''.join([itos[i] for i in l]) # Take a list of indices/integers, return a string

# Train and validation data
data = torch.tensor(encode(text), dtype=torch.long).to(device)
n = int(train_test_split * len(data)) # First n characters are for training
train_data = data[:n]
val_data = data[n:]

# Data Loading
def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Generates a tensor of shape (batch_size,) with random sequence start indices between 0 and len(data) - block_size
    x = torch.stack([data[i:i+block_size] for i in ix])       # stack all (ix holds batch_size many) sequences of this batch row-wise on top of each other to form a tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # same as x but shifted by one token
    x, y = x.to(device), y.to(device)
    return x, y  # x is batch_size x block_size, y is batch_size x block_size


@torch.no_grad() # Disable gradient calculation for this function
def evaluate_loss():
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to training mode
    return out


class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)      # Embedding the vocabulary, each individual token is represented by a vector of size vocab_size

    def forward(self, idx, targets=None):
        logits = self.embed(idx)                               # Embed the input indices, shape is now (batch_size, block_size, vocab_size) (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)                       # Transpose logits to (B, C, T) (B=batch_size, T=block_size, C=vocab_size)
            targets = targets.view(B*T)                        # Transpose targets to (B, T)
            loss = F.cross_entropy(logits, targets)                # Calculating cross entropy loss across all tokens in the batch
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                              # Forward pass (this is the forward function) with the current sequence of characters idx, results in (B, T, C)
            logits = logits[:, -1, :]                          # Focus on the last token from the logits (B, T, C) -> (B, C)
            probs = F.softmax(logits, dim=-1)                  # Calculate the set of probabilities for the next token based on this last token, results in (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # Sample the next token (B, 1), the token with the highest probability is sampled most likely
            idx = torch.cat((idx, idx_next), dim=1)            # Add the new token to the sequence (B, T+1) for the next iteration
        return idx  


# Model
model = BigramLM(vocab_size)
m = model.to(device)         # Move model parameters to device

# Create PyTorch Optimizer
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training
for iter in range(max_iters):
    xb, yb = get_batch('train', batch_size) # Get batch
    logits, loss = m(xb, yb)                # Forward pass
    loss.backward()                         # Backward pass
    opt.step()                        # Update parameters
    opt.zero_grad(set_to_none=True)   # Reset gradients

    if iter % eval_interval == 0:
        losses = evaluate_loss()
        print(f'Iter {iter:4d} | Train Loss {losses["train"]:6.4f} | Val Loss {losses["val"]:6.4f}')

# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with a zero context
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))