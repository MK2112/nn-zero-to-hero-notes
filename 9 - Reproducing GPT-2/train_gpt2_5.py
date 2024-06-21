import os
import sys
import math
import inspect
import time
from dataclasses import dataclass
import tiktoken
import numpy as np
from transformers import GPT2LMHeadModel
import torch
import torch.backends
import torch.nn as nn
import torch.nn.parallel.DistributedDataParallel as DDP
from hellaswag import iterate_examples, render_example
from torch.nn import functional as F
from torch.distributed import dist, init_process_group, destroy_process_group

# -----------------------------------------------------------------------------
# Run like so (specify the number of available GPUs):
# torchrun --standalone --nproc_per_node=[GPU COUNT] train_gpt2_5.py
# -----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token + padding for better performance
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension
    eval_iter: int = 250    # how often to evaluate the model

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
        x = x + self.attn(self.ln_1(x)) # (B, T, C); C = n_embd = 768
        x = x + self.mlp(self.ln_2(x))  # (B, T, C)
        return x # (B, T, C) still

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # n_layer decoder blocks
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
        # idx and targets are of shape (B, T)
        B, T = idx.size()
        # T now effectively holds largest sequence length in batch
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T), i.e. (seq_len)
        pos_emb = self.transformer.wpe(pos) # in shape: (seq_len); out shape: (seq_len, n_embd)
        tok_emb = self.transformer.wte(idx) # in shape: (batch_size, seq_len); out shape: (batch_size, seq_len, n_embd)
        x = tok_emb + pos_emb # add position embeddings to each sequence element of n_embd size
        # forward through the n_layer blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # output shape: (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
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
                # vanilla copy the other parameters
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
        # create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # See errata
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B # batch size
        self.T = T # context size
        self.process_rank = process_rank # rank of the current process
        self.num_processes = num_processes # total number of processes
        assert split in {'train', 'val'}, "split must be one of 'train' or 'val'"

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split '{split}' in '{data_root}'"
        if master_process:
            print(f"found {len(shards)} shards for split '{split}'")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # grab a chunk of tokens of size B * T + 1 (we explained this before)
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # input tensor of size (B * T)
        y = buf[1:].view(B, T)  # target tensor of size (B * T), right-shifted 1 position
        # advance position in data tensor
        # NEW: Jump to the next batch, taking into account the number of processes
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        # NEW: Check if our parallel-process-aware jump would be out of bounds
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            # reset position to beginning of data, w.r.t. process rank
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------

# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the B completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------

# Set up the DDP (distributed data parallel) environment
ddp = int(os.environ.get('RANK', -1)) != -1 # check if we are in a DDP environment/run
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "DDP requires CUDA for now - please run on a GPU node."
    init_process_group(backend='nccl')  # initialize distributed backend
    ddp_rank = int(os.environ['RANK'])  # rank of current process
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # local rank/GPU index of current process
    # (ddp_local_rank means within the node, ddp_rank is global, meaning across all nodes; node: machine with multiple GPUs (just saying))
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # number of processes in the DDP environment
    device = f"cuda:{ddp_local_rank}"  # map the device according to the local rank (indicates which GPU to use on the node)
    torch.cuda.set_device(device)      # set the device to the local rank
    master_process = ddp_rank == 0     # check if current process is master (master does logging, checkpointing etc.)
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

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
B = 16 # micro-batch size (I'll leave this at 16, you can set this lower or higher in powers of 2 as your hardware allows)
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "Batch size must be divisible by (micro-batch size * sequence length * ddp_world_size)."
# Accumulate gradients over grad_accum_steps instead of backpropagating each step
# Each process does B * T, and there are ddp_world_size many processes
# E.g. on 8 GPUs, we will now learn on 16 * 1024 * [8] = 131072 tokens per step before backpropagating
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    # Non-masters don't need to also print this
    print(f"Total desired batch size: {total_batch_size}")
    print(f"-> Calculated gradient accumulation steps: {grad_accum_steps}")

# DataLoaderLites for micro-batched training data and validation data respectively
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# TF32 tensor float precision for matmuls
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig()) # random weight initialization
model.to(device)

# [!!] Compiling would mess up our prescious sample generation in the training loop *and* the hellaswag eval 
# -> Deactivate it
use_compile = False
if use_compile and os.name == 'posix' and sys.platform != 'darwin':
    model = torch.compile(model) # compile model to TorchScript -> speed + memory savings
if ddp:
    # Once each GPUs backward is over, DDP will average the gradients across all GPUs
    model = DDP(model, device_ids=[ddp_local_rank])

# DDP requires a raw model reference for correct processing
raw_model = model.module if ddp else model

max_lr = 6e-4 # According to GPT-3 paper
min_lr = max_lr * 0.1
warmup_steps = 715    # Set for EDU_FineWeb10B, but in relation to dataset size like in GPT-3 paper
max_steps = 19073 * 4 # Set for EDU_FineWeb10B, but in relation to dataset size like in GPT-3 paper (takes 8h with 8 A100s)

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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and decays to zero
    return min_lr + coeff * (max_lr - min_lr)


# You can think of Adam as a combination of RMSprop and Stochastic Gradient Descent with momentum, i.e. a more sophisticated version of SGD.
# AdamW is a version of Adam that has a better implementation of weight decay. You can just use AdamW instead of Adam in most cases.
# https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
# For the sake of this example, we'll just use it and kind of treat is as a black box.
optimizer = raw_model.figure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
# Will contain train loss, val loss, hellaswag accuracies over training time
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    # Clearing the log file
    pass

# Load the tokenizer
enc = tiktoken.get_encoding('gpt2')

# Optimization loop
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1) # check if this is the last step
    # once in a while, check on the validation set
    if step % model.config.eval_iter == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.6f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.6f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # Optionally write out a checkpoint
                checkpoint_path = os.path.join(log_dir, f"checkpoint_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    # Added these to be able to easily resume training from a checkpoint
                    'optimizer': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state(),
                }
                torch.save(checkpoint, checkpoint_path)

    # once in a while, evaluate the hellaswag dataset
    if (step % model.config.eval_iter == 0 or last_step) and (not use_compile):
        model.eval()
        num_correct_norm = 0
        num_total = 0

        for i, example in enumerate(iterate_examples("val")):
            # Only process examples where i % ddep_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # Render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # Get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    logits, _ = model(tokens, mask)
                pred_norm = get_most_likely_row(tokens, mask, logits) # get the most likely completion per each sequence in the batch
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag acc_norm: {num_correct_norm}/{num_total}={acc_norm:.6f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hellaswag {acc_norm:.6f}\n")

    # once in a while, generate some text
    if ((step > 0 and step % model.config.eval_iter == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank) # seed the number generator for reproducibility
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1) # (B, T+1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    model.train()
    optimizer.zero_grad() # Reset gradients per macro-batch
    loss_accum = 0.0 # Loss accumulator over the micro-batches
    # Inner loop across micro-batches
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # (B, T)
        x, y = x.to(device), y.to(device)
        if ddp:
            # DDP will take care of averaging the loss across all GPUs
            # DDP will do so only after the last micro-batch in each accumulation cycle
            # Moved to here due to being required for both forward and backward pass alike
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # Normalize micro-batch loss by relation of full batch size to micro-batch size
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # This accumulates now, because we don't call zero_grad() in the inner loop
        loss.backward()
    if ddp:
        # Average the macro-batch loss across all GPUs, 
        # after having done that, synchronize this average loss to be the same value set across all GPUs
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
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
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size # Tokens processed per step across all GPUs
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            # Write train loss to file
            f.write(f"{step} train {loss_accum.item():.6f}\n")  

if ddp:
    destroy_process_group() # DDP cleanup

import sys; sys.exit(0)