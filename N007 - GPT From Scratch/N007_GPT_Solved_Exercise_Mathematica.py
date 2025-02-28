import torch
import random
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR


# tmux new -s gpt_math
# CUDA_VISIBLE_DEVICES=1 python 007_GPT_Solved_Exercise_Mathematica.py
#
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t gpt_math
# tmux list-sessions
# tmux kill-session -t gpt_math


# Hyperparameters
batch_size = 128    # How many independent sequences to process at once?
block_size = 27     # What is the maximum context length for predictions?
max_iters = 20_000  # How many training iterations to run?
eval_interval = 1_000    # How often to evaluate the model on the validation set?
learning_rate = 1e-3     # Learning rate for Adam optimizer (found through trial and error)
learning_rate_min = 1e-4 # Learning rate can hit this lower bound during scheduling
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Don't run on CPU if possible (it's really slow.)
eval_iters = 200    # How many batches to use per loss evaluation?
n_embd = 384        # Number of hidden units in the Transformer (384/6 = 64 dimensions per head)
n_head = 4          # Number of attention heads in a single Transformer layer
n_layer = 4         # Number of Transformer layers
dropout = 0.05      # Dropout probability

seed = 1337
torch.manual_seed(seed)
random.seed(seed)

def get_operation_weights(iteration):
    # Gradually introduce operations based on training progress
    progress = iteration / max_iters
    return {
        '+': 1.0,
        '-': max(0, min(1.0, progress * 6.0)),
        '*': max(0, min(1.0, progress * 5.5)), # trial and error to balance
        '/': max(0, min(1.0, progress * 3.5))
    }

def generate_problem(iteration=0):
    # Generate a mathematical problem with step-by-step solution
    weights = get_operation_weights(iteration)
    operations = []
    weights_list = []
    for op, weight in weights.items():
        if weight > 0:
            operations.append(op)
            weights_list.append(weight)
    
    operation = random.choices(operations, weights=weights_list)[0]
    a, b = random.randint(1, 9), random.randint(1, 9)
    
    # Calculate result and generate steps
    result = eval(f"{a}{operation}{b}" if operation != '/' else f"{a}//{b}")
    
    if operation in ['+', '-']:
        return f"{a}{operation}{b}=[{a}{operation}{b}={result}]={result};"
    elif operation == '*':
        # Show multiplication as repeated addition
        additions = [str(a)] * b
        steps = [f"{'+'.join(additions)}={result}"]
        return f"{a}{operation}{b}=[{'|'.join(steps)}]={result};"
    else: # division
        quotient = a // b
        return f"{a}{operation}{b}=[{a}//{b}={quotient}]={result};"

# Vocabulary with padding token
chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 
         '*', '/', '=', '[', ']', '|', 'r', ' ', ';', '#']
vocab_size = len(chars)

# Token weights for loss calculation
error_weights = torch.ones(vocab_size, dtype=torch.float32)
for i in range(10):  # Higher weights for numeric tokens
    error_weights[i] = 4.0

# Character to index mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos.get(i, '?') for i in l])

padding_token = stoi['#']

def get_batch(iteration=0):
    # Generate a batch of problems with their step-by-step solutions
    xb = torch.full((batch_size, block_size), padding_token, dtype=torch.long)
    yb = torch.full((batch_size, block_size), padding_token, dtype=torch.long)
    
    for i in range(batch_size):
        problem = generate_problem(iteration)
        split_point = problem.index('=')
        input_part = problem[:split_point+1]
        target_part = problem[split_point+1:]
        
        encoded_input = encode(input_part.ljust(block_size, '#'))
        encoded_target = encode(target_part.ljust(block_size, '#'))
        
        xb[i, :len(encoded_input)] = torch.tensor(encoded_input, dtype=torch.long)
        yb[i, :len(encoded_target)] = torch.tensor(encoded_target, dtype=torch.long)
    
    return xb.to(device), yb.to(device)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = num_heads * head_size
        
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.query = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, weight=error_weights.to(device), 
                                 ignore_index=padding_token)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize model and optimizer
model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=learning_rate_min)

print(f"Training on {device}")
print(f"{sum(p.numel() for p in model.parameters())/1e6:.4f}M parameters")

# Training loop
for iter in tqdm(range(max_iters)):
    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            losses = {'train': 0.0, 'val': 0.0}
            for _ in range(eval_iters):
                X, Y = get_batch(iter)
                _, loss = model(X, Y)
                losses['train'] += loss.item()
            losses['train'] /= eval_iters
            print(f"iter {iter}: loss {losses['train']:.4f}, lr {scheduler.get_last_lr()[0]:.7f}")
        model.train()
    
    xb, yb = get_batch(iter)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    scheduler.step()

torch.save(model, "gpt_mathematica.pt")

# Test the model
model.eval()
test_problems = [
    "2+3=",
    "5-2=",
    "4*3=",
    "8/4=",
]

print("\nTesting the model:")
with torch.no_grad():
    for problem in test_problems:
        context = torch.tensor([encode(problem)], dtype=torch.long, device=device)
        completion = decode(model.generate(context, max_new_tokens=block_size)[0].tolist())
        print(f"Input: {problem}")
        # Cut the completion at first ";"
        print(f"Generated: {completion.split(';')[0]}")
        expected = eval(problem[:-1])
        print(f"Expected: {expected}\n")