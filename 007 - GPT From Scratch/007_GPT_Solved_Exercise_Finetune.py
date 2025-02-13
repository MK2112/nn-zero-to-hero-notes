import torch
import random
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from torch.nn import functional as F
from transformers import AutoTokenizer


# tmux new -s gpt_finetune
# CUDA_VISIBLE_DEVICES=1 python 007_GPT_Solved_Exercise_Finetune.py
#
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t gpt_finetune
# tmux list-sessions
# tmux kill-session -t gpt_finetune
#
# 49.386577M parameter model


# Load train and validation dataset in streaming mode, without caching
hf_id = "Marcus2112/minipile_density-proportioned"
tokenizer_id = "gpt2"
dataset_train_size = 936_630 # Read that on huggingface.co/datasets/Marcus2112/minipile_density-proportioned
dataset_val_size = 9_366     # Same for this
streamed_train_dataset = load_dataset(hf_id, split="train", streaming=True, cache_dir=None)
streamed_val_dataset = load_dataset(hf_id, split="validation", streaming=True, cache_dir=None)

# Hyperparameters (same as for the tinyshakespeare-only attempt earlier, except accomodating for the larger dataset)
batch_size = 32  # How many independent sequences to process at once?
block_size = 256 # What is the maximum context length for predictions?
max_iters = dataset_train_size // batch_size # How many training iterations to run?
eval_interval = 1_000 # How often to evaluate the model on the validation set?
learning_rate = 3e-4  # Learning rate for Adam optimizer (found that through trial and error)
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Don't run on CPU if possible (it's slow. really.)
eval_iters = dataset_val_size // batch_size # How many validation iterations to run?
n_embd = 384     # Number of hidden units in the Transformer (384/6 = 64 dimensions per head)
n_head = 6       # Number of attention heads in a single Transformer layer
n_layer = 6      # Number of Transformer layers
dropout = 0.2    # Dropout probability

torch.manual_seed(1337)
print(f'Training on {device}')

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, cache_dir=None)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

# Create an iterator for the streamed dataset
train_dataset_iterator = iter(streamed_train_dataset)
validation_dataset_iterator = iter(streamed_val_dataset)

# Modify get_batch to tokenize on demand
def get_batch(split):
    global train_dataset_iterator, validation_dataset_iterator
    global streamed_train_dataset, streamed_val_dataset

    if split == 'train':
        dataset_iterator = train_dataset_iterator
    elif split == 'val':
        dataset_iterator = validation_dataset_iterator

    try:
        samples = [next(dataset_iterator) for _ in range(batch_size)]
    except StopIteration:
        if split == 'val':
            # Explicitly create new dataset
            streamed_val_dataset = load_dataset(hf_id, split="validation", streaming=True, cache_dir=None)
            validation_dataset_iterator = iter(streamed_val_dataset)
            dataset_iterator = validation_dataset_iterator
        elif split == 'train':
            # Explicitly create new dataset
            streamed_train_dataset = load_dataset(hf_id, split="train", streaming=True, cache_dir=None)
            train_dataset_iterator = iter(streamed_train_dataset)
            dataset_iterator = train_dataset_iterator

        # Try to get samples with the new iterator
        try:
            samples = [next(dataset_iterator) for _ in range(batch_size)]
        except Exception as e:
            print(f"Error after dataset reset: {str(e)}")
            raise

    tokenized = tokenizer([sample['text'] for sample in samples], truncation=True, 
                          padding='max_length', max_length=block_size, return_tensors='pt')
    x = tokenized['input_ids']
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = tokenizer.eos_token_id
    
    if x.size(1) != block_size:
        raise ValueError(f"Mismatch: Input size {x.size(1)} does not match block_size {block_size}. Check tokenization.")
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # Disable gradient calculation for this function
def estimate_loss():
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to training mode
    return out

class MultiHeadAttention(nn.Module):
    """ Multi-head self-attention processing all heads in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads           # Apply this many parallel attention layers
        self.head_size = head_size           # Each head has this size (part of the embedding size)
        self.n_embd = num_heads * head_size  # Total size of all heads together forms the token embedding size

        # Combining key, query, and value transformations across heads in a single linear layer each
        # All heads together process the input sequence and all together produce the output sequence
        # As self.embed = num_heads * head_size, input and output dim for all heads at once are the same (n_embd)
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.query = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Register a buffer so that causal mask is not a parameter of the model
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Final linear output transformation, dropout
        # Same as with the key, query, value transformations,
        # As self.embed = num_heads * head_size, input and output dim for all heads at once are the same (n_embd)
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # Batch size, sequence length, embedding size

        # Apply linear transformations to get keys, queries, and values for all heads
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)    # (B, T, C) -> (B, num_heads, T, head_size)
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, num_heads, T, head_size)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, num_heads, T, head_size)

        # Compute the attention scores
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5        # (B, num_heads, T, head_size) @ (B, num_heads, head_size, T) -> (B, num_heads, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Apply the causal mask
        wei = F.softmax(wei, dim=-1)  # Normalize attention scores to form (pseudo-)probabilities
        wei = self.dropout(wei)       # Apply dropout, promotes flexibility and robustness

        # Weighted aggregation of values
        out = wei @ v  # (B, num_heads, T, T) @ (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, num_heads, T, head_size) -> (B, T, C)

        # Final projection
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Linear layer with 4*n_embd outputs (AIAYN suggests 4*n_embd for residual connections as channel size)
            nn.ReLU(),                     # ReLU introduces non-linearity
            nn.Linear(4 * n_embd, n_embd), # Linear layer with n_embd outputs
            nn.Dropout(dropout),           # Dropout layer for regularization
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head                    # Adapting the head size to the number of heads
        self.sa = MultiHeadAttention(n_head, head_size) # Self-attention multi-head layer (the communication)
        self.ffwd = FeedFoward(n_embd)                  # Feed-forward so that the output has the same dimension as the input (the computation)
        self.ln1 = nn.LayerNorm(n_embd)                 # Layer normalization (normalizes the output of the self-attention layer)
        self.ln2 = nn.LayerNorm(n_embd)                 # Layer normalization (normalizes the output of the feed-forward layer)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # Residual connection, forking off to the self-attention layer, LayerNorm is applied before the self-attention layer
        x = x + self.ffwd(self.ln2(x)) # Residual connection, forking off to the feed-forward layer, LayerNorm is again applied before the feed-forward layer
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embd = nn.Embedding(vocab_size, n_embd)    # Embedding the vocabulary, each individual token is represented by a vector of size vocab_size x n_embd
        self.position_embd = nn.Embedding(block_size, n_embd) # Embedding the position, each position is represented by a vector of size block_size x n_embd
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # Linear layer to map the embedding to the vocabulary size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embd(idx)                              
        pos_embd = self.position_embd(torch.arange(T, device=device)) 
        x = tok_embd + pos_embd                                       
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                      

        if targets is None:
            loss = None
        else:
            # Flatten logits and targets
            logits = logits.view(-1, logits.size(-1)) # Shape: (B*T, vocab_size)
            targets = targets.view(-1)                # Shape: (B*T)
            # Mask out padding tokens
            pad_mask = targets != tokenizer.pad_token_id
            logits = logits[pad_mask]    # Keep only non-pad logits
            targets = targets[pad_mask]  # Keep only non-pad targets
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]                    # Condition on the last block_size tokens (B, T)
            logits, _ = self(idx_cond)                         # Forward pass (this is the forward function) with the current sequence of characters idx, results in (B, T, C)
            logits = logits[:, -1, :]                          # Focus on the last token from the logits (B, T, C) -> (B, C)
            probs = F.softmax(logits, dim=-1)                  # Calculate the set of probabilities for the next token based on this last token, results in (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # Sample the next token (B, 1), the token with the highest probability is sampled most likely
            idx = torch.cat((idx, idx_next), dim=1)            # Add the new token to the sequence (B, T+1) for the next iteration
        return idx

# Model
model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') # print the number of parameters in the model

# Create a PyTorch optimizer
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for itr in tqdm(range(max_iters)):
    if itr % eval_interval == 0 or itr == max_iters - 1:
        losses = estimate_loss()
        print(f"step {itr}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')     # Get batch
    logits, loss = model(xb, yb)    # Forward pass
    opt.zero_grad(set_to_none=True) # Reset gradients
    loss.backward()                 # Backward pass
    opt.step()                      # Update parameters

torch.save(model, "gpt_mini.pt")

# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with single token as context
print(tokenizer.decode(m.generate(context, max_new_tokens=block_size)[0].tolist(), skip_special_tokens=True))

# Commence the finetuning
with open("../tiny-shakespeare.txt", "r") as f:
    shakespeare_text = f.read()

# Tokenize the Shakespeare text
def tokenize_text(text, tokenizer, block_size):
    tokens = tokenizer(text, truncation=False, padding=False, return_tensors="pt")
    input_ids = tokens['input_ids'].squeeze(0)
    # Split into chunks of block_size
    chunks = input_ids.split(block_size)
    if len(chunks[-1]) < block_size:
        chunks = chunks[:-1]
    return torch.stack(chunks)

# Prepare train and validation datasets
shakespeare_tokens = tokenize_text(shakespeare_text, tokenizer, block_size)
train_size = int(0.8 * len(shakespeare_tokens))
train_data = shakespeare_tokens[:train_size]
val_data = shakespeare_tokens[train_size:]

finetune_batch_size = 32
def fine_tune_data_iterator(data, batch_size):
    for i in range(0, len(data), batch_size):
        x = data[i:i + batch_size, :]
        y = x.clone()
        y[:, :-1] = x[:, 1:]
        y[:, -1] = tokenizer.eos_token_id
        yield x.to(device), y.to(device)

# Finetuning hyperparameters
finetune_lr = 2e-5
finetune_iters = 5_000
finetune_eval_interval = 1_000
finetune_eval_iters = len(val_data) // finetune_batch_size

# Finetuning optimizer
finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)

def finetune_estimate_loss():
    """Estimate the loss during fine-tuning."""
    out = {}
    model.eval()
    for split, data in zip(['train', 'val'], [train_data, val_data]):
        losses = []
        iterator = fine_tune_data_iterator(data, finetune_batch_size)
        for X, Y in iterator:
            logits, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# Fine-tuning loop
print("Starting fine-tuning...")
for itr in tqdm(range(finetune_iters)):
    if itr % finetune_eval_interval == 0 or itr == finetune_iters - 1:
        losses = finetune_estimate_loss()
        print(f"Fine-tuning step {itr}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Fine-tuning batch
    finetune_iterator = fine_tune_data_iterator(train_data, finetune_batch_size)
    try:
        xb, yb = next(finetune_iterator)
    except StopIteration:
        finetune_iterator = fine_tune_data_iterator(train_data, finetune_batch_size)
        xb, yb = next(finetune_iterator)

    logits, loss = model(xb, yb)
    finetune_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    finetune_optimizer.step()

# Save the fine-tuned model
torch.save(model, "gpt_mini_shakespeare.pt")

# Generate text from the fine-tuned model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with single token as context
print("\nGenerated text after fine-tuning:")
print(tokenizer.decode(model.generate(context, max_new_tokens=block_size)[0].tolist(), skip_special_tokens=True))
