import os
import sys
import math
import time
import torch
import inspect
import torch.backends
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F

# -----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # Love how this looks lol
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        
        # Verbose Attention implementation
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Flash Attention implementation we use instead (might throw a warning on Win32)
        # https://www.reddit.com/r/comfyui/comments/1cerq2e/is_uh_comfyanon_aware_that_pytorch_flash/
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, shape is (B, T, C)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        # essentially iterates over submodules and applies the function to them
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Linear weights distributed with mean 0 and std 0.02
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 1 / sqrt(N) scaling fo the std dev
                # 2 * n_layer because we have attention *and* mlp per block (blocks counted by n_layer)
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                # Biases explicitly initialized to 0
                torch.nn.init.zeros_(module.bias)
        # Embedding weights distributed with mean 0 and std 0.02
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx, targets both of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a HuggingFace/Transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the OpenAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, but all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = torch.cuda.is_available() and fused_available and device.startswith('cuda')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B # batch size
        self.T = T # context size

        # at init load tokens from disk and store them in memory
        with open('../tiny-shakespeare.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text) # encode full text into tokens
        self.tokens = torch.tensor(tokens) # wrap with tensor
        self.tok_count = len(self.tokens)
        # Just some stats for us nerds
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # token-level start position for each next batch
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # grab a chunk of tokens of size B * T + 1 (we explained this before)
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # input tensor of size (B * T)
        y = buf[1:].view(B, T)  # target tensor of size (B * T), right-shifted 1 position
        # advance position in data tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) >= len(self.tokens):
            # reset position to beginning of data
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------

# Find the best available device to train on
device = "cpu"
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_initialized():
    device = "mps" # Apple Silicon
print(f"Using device: {device}")

# Reproducibility
torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19, ~0.5M tokens, as per GPT-3 paper
# Doing 16384 tokens per micro-batch 
# -> 32 (grad_accum_steps) micro-batches to accumulate into a full batch
B = 16 # micro-batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "Batch size must be divisible by (micro-batch size * sequence length)."
# Accumulate gradients over grad_accum_steps instead of backpropagating each step
grad_accum_steps = total_batch_size // (B * T) # 32 micro-batches making up 1 macro-batch
print(f"Total desired batch size: {total_batch_size}")
print(f"-> Calculated gradient accumulation steps: {grad_accum_steps}")

# DataLoaderLite for micro-batched training data
train_loader = DataLoaderLite(B=T, T=B)

# TF32 tensor float precision for matmuls
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig()) # random weight initialization
model.to(device)

# Check if we run on Linux:
if os.name == 'posix' and sys.platform != 'darwin':
    model = torch.compile(model) # compile model to TorchScript -> speed + memory savings
else:
    print("[!] Not running Linux - Skipping platform-unsupported torch.compile()")


max_lr = 6e-4 # According to GPT-3 paper
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it < warmup_steps:
        # 1) Linear warmup region for warmup_iters steps
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        # 2) if it > lr_decay_iters, flat out return the min_lr
        return min_lr
    # 3) In between warmup and max_steps, cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to zero
    return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
# You can think of Adam as a combination of RMSprop and Stochastic Gradient Descent with momentum, i.e. a more sophisticated version of SGD.
# AdamW is a version of Adam that has a better implementation of weight decay. You can just use AdamW instead of Adam in most cases.
# https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
# For the sake of this example, we'll just use it and kind of treat is as a black box.

# Optimization loop
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() # Reset gradients per macro-batch
    loss_accum = 0.0 # Loss accumulator over the micro-batches
    # Inner loop across micro-batches
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # (B, T)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # Normalize micro-batch loss by relation of full batch size to micro-batch size
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # This accumulates now, because we don't call zero_grad() in the inner loop
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping the global norm at 1.0
    # Determine and set the learning rate for this iteration step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        # Kind of side-injecting the actual learning rate we want to apply to the parameters here
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for GPU to finish above scheduled workload
    t1 = time.time()
    dt = (t1 - t0) # millisecond time difference
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps # Tokens processed per step
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")      

import sys; sys.exit(0)

# generate! right now x is (B, T) where B = 5, T = 8
model.eval()
num_return_sequences = 5
max_length = 30
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1) # (B, T+1)

# print the generated text
for step in range(num_return_sequences):
    tokens = x[step, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
