"GPT modelling copied from https://github.com/karpathy/nanochat/tree/master"

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.act_block import AdaptiveBlock
from models.ponder_block import AdaptivePonderBlock



@dataclass
class TransformerBlockConfig:
    sequence_len: int = 128
    vocab_size: int = 16
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 256
    n_layer: int = 6

    # share_kv
    share_kv: bool = False
    share_kv_n_layers: int = 2

    #recursive
    recursive: bool = False
    recursion_depth: int = 5

    # Adaptive computation parameters (ACT)
    use_adaptive_computation: bool = True
    n_layers_per_block: int = 1  # Number of layers per adaptive block
    max_pondering_steps: int = 5
    act_threshold: float = 0.99
    halting_penalty: float = 0.01  # tau in ACT paper

    # --- PonderNet (probabilistic halting) ---
    use_pondernet: bool = False      # turn on to use PonderNet instead of ACT
    geom_lambda_prior: float = 0.5   # λ for geometric prior (E[T]=1/λ)
    kl_weight_beta: float = 0.05     # β weight on KL(p||q)
    cdf_early_stop: float = 0.999    # optional early stop when CDF≈1



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



def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out


class HaltingUnit(nn.Module):
    """Computes halting probability for adaptive computation."""
    def __init__(self, n_embd):
        super().__init__()
        self.halting_linear = nn.Linear(n_embd, 1, bias=True)
        nn.init.constant_(self.halting_linear.bias, 0.0)
        self.halting_linear.is_halting_unit = True

    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd)
        Returns:
            halting_prob: (B, T, 1) - probability of halting at this step
        """
        return torch.sigmoid(self.halting_linear(x))


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

    def forward(self, x, cos_sin, kv_cache, shared_kv=None):
        """
        Args:
            x: input (B, T, C)
            cos_sin: rotary_embd (full cache)
            kv_cache: (for inference)
            shared_kv: kv from first recursion step for adaptive computation
        """
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        # Apply rotary embeddings
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        q = norm(q)
        q = q.transpose(1, 2)

        #  on ith recursion step(i != 1)
        if shared_kv is not None:
            k, v = shared_kv
        else:
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
        
        # Return k, v for potential sharing in adaptive computation
        return y, (k, v) if shared_kv is None else None

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

    def forward(self, x, cos_sin, kv_cache, shared_kv=None):
        """
        Standard forward pass through one transformer layer.
        
        Args:
            x: input (B, T, C)
            cos_sin: rotary embeddings
            kv_cache: KV cache for inference
            shared_kv: Optional (k, v) from first recursion for adaptive computation
            input_pos: Optional position indices for RoPE
        
        Returns:
            x: output after attention + MLP
            new_kv: K,V pair from attention (or None if using shared_kv)
        """
        attn_out, new_kv = self.attn(norm(x), cos_sin, kv_cache, shared_kv)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x, new_kv





class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Build adaptive blocks or individual layers based on config
        if config.use_adaptive_computation:
            assert config.n_layer % config.n_layers_per_block == 0, \
                f"n_layer ({config.n_layer}) must be divisible by n_layers_per_block ({config.n_layers_per_block})"
            n_blocks = config.n_layer // config.n_layers_per_block
            blocks = []
            for block_idx in range(n_blocks):
                start_layer_idx = block_idx * config.n_layers_per_block
                blocks.append(AdaptiveBlock(config, start_layer_idx, config.n_layers_per_block))
            self.transformer = nn.ModuleDict({
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(blocks),
            })
        elif config.use_pondernet:
            assert config.n_layer % config.n_layers_per_block == 0
            n_blocks = config.n_layer // config.n_layers_per_block
            blocks = []
            for block_idx in range(n_blocks):
                start_layer_idx = block_idx * config.n_layers_per_block
                blocks.append(AdaptivePonderBlock(config, start_layer_idx, config.n_layers_per_block))
            self.transformer = nn.ModuleDict({
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(blocks),
            })
        else:
            # standard transformer
            if config.share_kv:
                assert config.n_layer % config.share_kv_n_layers == 0, \
                f"n_layer ({config.n_layer}) must be divisible by share_kv_n_layers ({config.share_kv_n_layers})"

            if config.recursive:
                assert config.recursion_depth > 0, "Recursion depth must be positive"

            self.transformer = nn.ModuleDict({
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
            })
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 4
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            if isinstance(block, AdaptiveBlock):
                for layer in block.layers:
                    nn.init.zeros_(layer.mlp.c_proj.weight)
                    nn.init.zeros_(layer.attn.c_proj.weight)
            else:
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        #if self.transformer.wte.weight.device.type == "cuda":
            #self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None and not hasattr(module, 'is_halting_unit'):
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
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
        #cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device


    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.02, matrix_lr=0.02, halting_lr=0.05, weight_decay=0.0):
        """
        Setup optimizers for training. 
        Note: Requires external optimizers (Muon, DistAdamW, DistMuon) to be imported if using DDP.
        For simple training, just use standard PyTorch optimizers.
        """
        model_dim = self.config.n_embd
        
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = []
        halting_params = []
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        for block in self.transformer.h:
            if isinstance(block, AdaptiveBlock):
                # Add halting unit params separately
                halting_params.extend(block.halting_unit.parameters())
                # Add layer params (excluding halting)
                for layer in block.layers:
                    matrix_params.extend(layer.parameters())
            elif isinstance(block, AdaptivePonderBlock):
                halting_params.extend(block.halting_unit.parameters())
                for layer in block.layers:
                    matrix_params.extend(layer.parameters())
            elif isinstance(block, AdaptivePonderBlock):
                halting_params.extend(block.halting_unit.parameters())
                for layer in block.layers:
                    matrix_params.extend(layer.parameters())


                matrix_params.extend(block.parameters())
        
        # Create a simple AdamW optimizer for all parameters
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        param_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=matrix_params, lr=matrix_lr * dmodel_lr_scale),
            dict(params=halting_params, lr=halting_lr * dmodel_lr_scale),
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', ponder_mask=None):
        B, T = idx.size()

        if ponder_mask is not None and ponder_mask.dim() == 2:              # (B, T) -> (B, T, 1)
            ponder_mask = ponder_mask.unsqueeze(-1)

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        #assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        if self.config.use_adaptive_computation:
            return self.forward_adaptive(idx, targets, cos_sin,  kv_cache, loss_reduction, ponder_mask)
        elif self.config.use_pondernet:
            return self.forward_pondernet(idx, targets, cos_sin, kv_cache, loss_reduction, ponder_mask)

        # Standard forward pass (no adaptive computation)
        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        
        shared_kv = None
        for i, layer in enumerate(self.transformer.h):
            
            if self.config.recursive:
                for j in range(self.config.recursion_depth):
                    x, shared_kv = layer(x, cos_sin, kv_cache, shared_kv=shared_kv)
                shared_kv = None

            elif self.config.share_kv and i % self.config.share_kv_n_layers == 0:
                x, shared_kv = layer(x, cos_sin, kv_cache, shared_kv=shared_kv)

            else:
                x, _ = layer(x, cos_sin, kv_cache, shared_kv=shared_kv)

        x = norm(x)

        # Forward the lm_head (compute logits)
        #softcap = 15

        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            #logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss, None
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            #logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    def forward_adaptive(self, idx, targets,  cos_sin, kv_cache, loss_reduction='mean', ponder_mask=None):
        B, T = idx.size()
        config = self.config

        x = self.transformer.wte(idx)
        x = norm(x)

        total_act_ponder = torch.zeros(B, T, 1, device=idx.device)
        total_expected   = torch.zeros(B, T, 1, device=idx.device) 

        for block in self.transformer.h:
            x, act_ponder, n_steps = block(x, cos_sin, kv_cache, ponder_mask=ponder_mask)
            total_act_ponder = total_act_ponder + act_ponder
            total_expected   = total_expected   + n_steps

        x = norm(x)

        #softcap = 15
        logits = self.lm_head(x)
        #logits = softcap * torch.tanh(logits / softcap)
        logits = logits.float()

        if ponder_mask is None:
            act_penalty =  self.config.halting_penalty * total_act_ponder.mean()
            self.last_expected_steps = total_expected.mean().item() / self.config.n_layer
        else:
            pm = ponder_mask.float()
            denom = pm.sum().clamp_min(1.0) 
            act_penalty = (self.config.halting_penalty * (total_act_ponder * pm).sum()) / denom
            self.last_expected_steps = (total_expected * pm).sum().item() / (denom * self.config.n_layer)

        if targets is None:
            return logits
        task_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
       

        self.last_act_penalty = act_penalty.item()
        return task_loss, act_penalty
    
    def forward_ponder(self, idx, targets, cos_sin, kv_cache, loss_reduction='mean', ponder_mask=None):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        x = norm(x)

        total_kl = torch.zeros((), device=idx.device)
        total_expected = torch.zeros((), device=idx.device)

        for block in self.transformer.h:  # these are AdaptivePonderBlock
            x, kl_term, exp_steps = block(x, cos_sin, kv_cache, ponder_mask=ponder_mask)
            total_kl = total_kl + kl_term
            total_expected = total_expected + exp_steps.mean()

        x = norm(x)
        logits = self.lm_head(x).float()

        # no ACT penalty here; use KL instead
        if targets is None:
            # stash metrics for generate()
            self.last_expected_steps = (total_expected.item() / self.config.n_layer)
            self.last_ponder_kl = total_kl.item()
            return logits

        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )

        loss = ce + self.config.kl_weight_beta * total_kl

        # metrics (normalize per-layer for parity with ACT scalar you expose)
        self.last_expected_steps = (total_expected.item() / self.config.n_layer)
        self.last_ponder_kl = total_kl.item()
        return loss, None  # or return aux dict



    @torch.inference_mode()
    def generate(self, tokens, max_tokens, ponder_mask=None, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        if temperature > 0:
            rng = torch.Generator(device=device).manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        if ponder_mask:
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
        logits = self.forward(ids, kv_cache=kv_cache, ponder_mask=ponder_mask)
        
        for _ in range(max_tokens):
            logits_last = logits[:, -1, :]

            if top_k is not None:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = -float('inf')
            
            if temperature > 0:
                probs = F.softmax(logits_last / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits_last, dim=-1, keepdim=True)
                

            yield next_id.item(), getattr(self, "last_expected_steps", float("nan"))

            logits = self.forward(next_id, kv_cache=kv_cache)
