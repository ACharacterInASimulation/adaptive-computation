import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerBlockConfig:
    sequence_len: int = 128
    vocab_size: int = 16
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 256
    n_layer: int = 6
    # Early exit parameters
    n_layers_per_block: int = 2  # Number of layers per block
    use_early_exit: bool = True
    exit_threshold: float = 0.8  # Confidence threshold for early exit
    exit_penalty: float = 0.01  # Penalty for encouraging earlier exits


class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0 # current position in time in the cache

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                # num_layers, batch_size, num_heads, head_dim must match
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size can be expanded
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len: self must be longer than other
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # as much as we need plus buffer of 1024
            t_needed = (t_needed + 1023) & ~1023 # then round up to the nearest multiple of 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# def norm(x):
#     # Purely functional rmsnorm with no learnable params
#     return F.rms_norm(x, (x.size(-1),))

def norm(x):
    # Purely functional rmsnorm with no learnable params
    # Manual implementation for PyTorch < 2.1
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + 1e-5)
    return x

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out


class ExitHead(nn.Module):
    """Exit classification head for early exit."""
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.exit_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd)
        Returns:
            logits: (B, T, vocab_size)
        """
        return self.exit_head(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        """
        Args:
            x: input (B, T, C)
            cos_sin: rotary_embd (full cache)
            kv_cache: (for inference)
        """
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        # Apply rotary embeddings
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        q = norm(q)
        q = q.transpose(1, 2)

        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        k = apply_rotary_emb(k, cos, sin)
        k = norm(k)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Single transformer layer (attention + MLP)."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        """
        Standard forward pass through one transformer layer.

        Args:
            x: input (B, T, C)
            cos_sin: rotary embeddings
            kv_cache: KV cache for inference

        Returns:
            x: output after attention + MLP
        """
        attn_out = self.attn(norm(x), cos_sin, kv_cache)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x


class TransformerBlock(nn.Module):
    """
    Block containing N transformer layers with optional early exit capability.
    Each layer can have an exit head for per-layer early exit within the block.
    Early exit is controlled by ponder_mask - only tokens with ponder_mask=1 can exit early.
    """
    def __init__(self, config, start_layer_idx, n_layers_per_block, is_last_block=False):
        super().__init__()
        self.config = config
        self.start_layer_idx = start_layer_idx
        self.n_layers = n_layers_per_block
        self.is_last_block = is_last_block

        # Create N transformer layers
        self.layers = nn.ModuleList([
            Block(config, start_layer_idx + i)
            for i in range(n_layers_per_block)
        ])

        # Create exit heads for each layer in the block (except last layer if last block)
        self.exit_heads = nn.ModuleList()
        if config.use_early_exit:
            for i in range(n_layers_per_block):
                # Skip exit head for last layer of last block
                if is_last_block and i == n_layers_per_block - 1:
                    self.exit_heads.append(None)
                else:
                    self.exit_heads.append(ExitHead(config.n_embd, config.vocab_size))
        else:
            self.exit_heads = [None] * n_layers_per_block

    def forward(self, x, cos_sin, kv_cache=None, use_early_exit=False, ponder_mask=None):
        """
        Forward pass through layers in this block with optional per-layer early exit.
        Early exit only applies to tokens where ponder_mask=1.

        Args:
            x: input (B, T, C)
            cos_sin: rotary embeddings
            kv_cache: KV cache for inference
            use_early_exit: whether to check for early exit at each layer
            ponder_mask: (B, T) or (B, T, 1) mask indicating which tokens can use early exit

        Returns:
            x: output after processing
            layer_exit_info: list of (layer_idx_in_block, exit_logits, exited_mask) tuples
        """
        B, T, C = x.size()
        layer_exit_info = []

        # Process ponder_mask
        if ponder_mask is not None:
            if ponder_mask.dim() == 2:
                ponder_mask = ponder_mask.unsqueeze(-1)  # (B, T) -> (B, T, 1)
            can_exit = ponder_mask.squeeze(-1).bool()  # (B, T)
        else:
            can_exit = torch.ones(B, T, dtype=torch.bool, device=x.device)

        # Track which tokens have exited within this block
        exited_in_block = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        for layer_idx, (layer, exit_head) in enumerate(zip(self.layers, self.exit_heads)):
            # Process layer for all tokens (even exited ones, for gradient flow)
            x_new = layer(x, cos_sin, kv_cache)

            # Only update x for tokens that haven't exited yet
            if use_early_exit and exited_in_block.any():
                x = torch.where(exited_in_block.unsqueeze(-1), x, x_new)
            else:
                x = x_new

            # Check for early exit at this layer
            exit_logits = None
            if exit_head is not None:
                exit_logits = exit_head(norm(x))

                if use_early_exit:
                    with torch.no_grad():
                        probs = F.softmax(exit_logits.float(), dim=-1)
                        max_probs = probs.max(dim=-1)[0]  # (B, T)
                        # Only allow exit for tokens that:
                        # 1. Can ponder (ponder_mask=1)
                        # 2. Haven't exited yet
                        # 3. Meet confidence threshold
                        should_exit = (max_probs >= self.config.exit_threshold) & can_exit & (~exited_in_block)

                        if should_exit.any():
                            exited_in_block = exited_in_block | should_exit
                            layer_exit_info.append((layer_idx, exit_logits, should_exit))
                        else:
                            layer_exit_info.append((layer_idx, exit_logits, None))
                else:
                    layer_exit_info.append((layer_idx, exit_logits, None))

        return x, layer_exit_info


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build blocks
        assert config.n_layer % config.n_layers_per_block == 0, \
            f"n_layer ({config.n_layer}) must be divisible by n_layers_per_block ({config.n_layers_per_block})"
        n_blocks = config.n_layer // config.n_layers_per_block
        blocks = []
        for block_idx in range(n_blocks):
            start_layer_idx = block_idx * config.n_layers_per_block
            is_last = (block_idx == n_blocks - 1)
            blocks.append(TransformerBlock(config, start_layer_idx, config.n_layers_per_block, is_last))

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList(blocks),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 4
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Track early exit stats
        self.last_exit_distribution = None

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights and exit heads in all blocks
        for block in self.transformer.h:
            for layer in block.layers:
                nn.init.zeros_(layer.mlp.c_proj.weight)
                nn.init.zeros_(layer.attn.c_proj.weight)
            if hasattr(block, 'exit_heads'):
                for exit_head in block.exit_heads:
                    if exit_head is not None:
                        nn.init.zeros_(exit_head.exit_head.weight)

        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.02, matrix_lr=0.02, exit_lr=0.02, weight_decay=0.0):
        """
        Setup optimizers for training.
        """
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = []
        exit_params = []
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        for block in self.transformer.h:
            # Add layer params
            for layer in block.layers:
                matrix_params.extend(layer.parameters())
            # Add exit head params separately
            if hasattr(block, 'exit_heads'):
                for exit_head in block.exit_heads:
                    if exit_head is not None:
                        exit_params.extend(exit_head.parameters())

        # Scale the LR for the AdamW parameters by √1/√dmodel
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling the LR for the AdamW parameters √1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=matrix_params, lr=matrix_lr * dmodel_lr_scale),
        ]

        if exit_params:
            param_groups.append(dict(params=exit_params, lr=exit_lr * dmodel_lr_scale))

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )

        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_exit_info=False, 
                use_early_exit_training=None, ponder_mask=None):
        B, T = idx.size()

        if use_early_exit_training is None:
            use_early_exit_training = self.config.use_early_exit

        # Process ponder_mask
        if ponder_mask is not None and ponder_mask.dim() == 2:
            ponder_mask = ponder_mask.unsqueeze(-1)  # (B, T) -> (B, T, 1)

        # Grab the rotary embeddings for the current sequence length
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"

        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)

        # Track exit information: (block_idx, layer_idx_in_block, exit_logits, exited_mask)
        all_exit_info = []

        for block_idx, block in enumerate(self.transformer.h):
            x, layer_exit_info = block(x, cos_sin, kv_cache, 
                                      use_early_exit=use_early_exit_training and targets is not None,
                                      ponder_mask=ponder_mask)

            # Store exit info with block context
            for layer_idx_in_block, exit_logits, exited_mask in layer_exit_info:
                global_layer_idx = block_idx * self.config.n_layers_per_block + layer_idx_in_block
                all_exit_info.append((block_idx, layer_idx_in_block, global_layer_idx, exit_logits, exited_mask))

        x = norm(x)

        # Forward the lm_head (compute logits)
        logits = self.lm_head(x)
        logits = logits.float() # use tf32/fp32 for logits

        if targets is not None:
            # Training mode: compute and return the loss

            # Build exit tracking: which layer did each token last exit at?
            exit_layer = torch.full((B, T), self.config.n_layer, dtype=torch.long, device=idx.device)  # default: no early exit

            for block_idx, layer_in_block, global_layer, exit_logits, exited_mask in all_exit_info:
                if exited_mask is not None and exited_mask.any():
                    # Only update if this is an earlier exit than previously recorded
                    exit_layer = torch.where(
                        exited_mask & (exit_layer == self.config.n_layer),
                        torch.full_like(exit_layer, global_layer),
                        exit_layer
                    )

            # Compute main task loss
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )

            # Compute auxiliary exit losses (only for tokens that can ponder)
            exit_loss = 0.0
            exit_penalty = 0.0

            if self.config.use_early_exit and all_exit_info:
                total_exit_loss = 0.0
                num_exit_heads = 0

                # Create mask for ponderable tokens
                if ponder_mask is not None:
                    can_ponder = ponder_mask.squeeze(-1).bool()  # (B, T)
                else:
                    can_ponder = torch.ones(B, T, dtype=torch.bool, device=idx.device)

                for block_idx, layer_in_block, global_layer, exit_logits, exited_mask in all_exit_info:
                    if exit_logits is not None:
                        # Compute loss for this exit head (only on ponderable tokens)
                        exit_logits_float = exit_logits.float()
                        head_loss = F.cross_entropy(
                            exit_logits_float.view(-1, exit_logits_float.size(-1)),
                            targets.reshape(-1),
                            ignore_index=-1,
                            reduction=loss_reduction
                        )
                        total_exit_loss += head_loss
                        num_exit_heads += 1

                        # Add penalty based on layer depth (only for ponderable tokens that exited)
                        if exited_mask is not None and exited_mask.any():
                            mask = exited_mask & can_ponder & (targets != -1)
                            if mask.any():
                                ponder_count = (can_ponder & (targets != -1)).sum().float()
                                if ponder_count > 0:
                                    weight = mask.sum().float() / ponder_count
                                    exit_penalty += (global_layer + 1) * weight * self.config.exit_penalty

                exit_loss = total_exit_loss / num_exit_heads if num_exit_heads > 0 else 0.0

            total_loss = task_loss + exit_loss + exit_penalty

            # Store stats
            self.last_exit_loss = exit_loss if isinstance(exit_loss, float) else exit_loss.item()
            self.last_exit_penalty = exit_penalty if isinstance(exit_penalty, float) else exit_penalty.item()

            if return_exit_info:
                # Compute exit distribution per layer for this batch (only for ponderable tokens)
                if ponder_mask is not None:
                    mask = (targets != -1) & ponder_mask.squeeze(-1).bool()
                else:
                    mask = targets != -1
                    
                if mask.any():
                    exit_distribution = []
                    # Count exits at each layer
                    for layer_idx in range(self.config.n_layer + 1):
                        if layer_idx == self.config.n_layer:
                            # Final layer (no early exit)
                            count = ((exit_layer == self.config.n_layer) & mask).sum().item()
                        else:
                            count = ((exit_layer == layer_idx) & mask).sum().item()
                        exit_distribution.append(count)
                    self.last_exit_distribution = exit_distribution

            return total_loss
        else:
            # Inference mode: compute and return the logits
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, ponder_mask=None, temperature=1.0, top_k=None, seed=42, use_early_exit=None):
        """
        Generate tokens with optional early exit during inference.
        Early exit only applies to tokens where ponder_mask=1.

        Args:
            tokens: list of input token ids
            max_tokens: maximum number of tokens to generate
            ponder_mask: list indicating which positions can use early exit (1=can exit, 0=must use all layers)
            temperature: sampling temperature
            top_k: top-k sampling
            seed: random seed
            use_early_exit: If True, exit early when confidence threshold is met.
                          If None, use config.use_early_exit
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        if temperature > 0:
            rng = torch.Generator(device=device).manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        if use_early_exit is None:
            use_early_exit = self.config.use_early_exit

        # Convert ponder_mask to tensor if provided
        if ponder_mask is not None:
            ponder_mask = torch.tensor([ponder_mask], dtype=torch.float32, device=device)

        head_dim = self.config.n_embd // self.config.n_head
        kv_cache = KVCache(
            batch_size=1,
            num_heads=self.config.n_kv_head,
            seq_len=max(len(tokens) + max_tokens, self.config.sequence_len),
            head_dim=head_dim,
            num_layers=self.config.n_layer,
        )

        # Prefill
        _ = self.forward(ids, kv_cache=kv_cache, ponder_mask=ponder_mask)

        exit_counts = [0] * (self.config.n_layer + 1)  # Track which layer we exit at

        for _ in range(max_tokens):
            last = ids[:, -1:]

            if use_early_exit and ponder_mask is not None and ponder_mask[:, -1].item() == 1:
                # Manual forward pass with early exit logic for ponderable tokens
                B, T = last.size()
                T0 = kv_cache.get_pos()
                cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

                x = self.transformer.wte(last)
                x = norm(x)

                exited = False
                exit_layer_idx = None

                for block_idx, block in enumerate(self.transformer.h):
                    for layer_idx, (layer, exit_head) in enumerate(zip(block.layers, block.exit_heads)):
                        x = layer(x, cos_sin, kv_cache)

                        if exit_head is not None and not exited:
                            exit_logits = exit_head(norm(x))
                            exit_probs = F.softmax(exit_logits.float() / temperature, dim=-1)
                            max_prob = exit_probs.max().item()

                            if max_prob >= self.config.exit_threshold:
                                logits = exit_logits[:, -1, :]
                                exited = True
                                exit_layer_idx = block_idx * self.config.n_layers_per_block + layer_idx
                                exit_counts[exit_layer_idx] += 1
                                break

                    if exited:
                        break

                if not exited:
                    x = norm(x)
                    logits = self.lm_head(x)[:, -1, :]
                    exit_counts[-1] += 1
            else:
                # Standard forward pass (no early exit for non-ponderable tokens)
                logits = self.forward(last, kv_cache=kv_cache, ponder_mask=ponder_mask if ponder_mask is not None else None)
                logits = logits[:, -1, :]
                exit_counts[-1] += 1

            # Sample next token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_id), dim=1)
            
            # Update ponder_mask for generated token (always allow pondering for generated tokens)
            if ponder_mask is not None:
                new_ponder = torch.ones(1, 1, dtype=torch.float32, device=device)
                ponder_mask = torch.cat([ponder_mask, new_ponder], dim=1)
            
            # Yield token and average exit layer
            if exit_counts[-1] > 0:
                total_exits = sum(exit_counts)
                avg_layer = sum(i * count for i, count in enumerate(exit_counts)) / total_exits if total_exits > 0 else self.config.n_layer
                yield next_id.item(), avg_layer
            else:
                yield next_id.item(), float('nan')

        # Store exit distribution for analysis
        self.last_exit_distribution = exit_counts