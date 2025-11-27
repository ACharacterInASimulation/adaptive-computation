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
    exit_penalty: float = 0.01   # Penalty for encouraging earlier exits


class KVCache:
    """
    KV cache for a stack of layers.
    NOTE: Here, a KVCache instance serves one block (num_layers = n_layers_per_block).
    Its .pos advances automatically after the *last layer in this cache* inserts.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # shape: (L, 2, B, H, T, D), where L = num_layers for this cache
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0  # time pointer for this cache

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def advance(self, T_add: int = 1):
        """Manually advance the time pointer (useful if a block exits before its last layer writes)."""
        self.pos += T_add

    def prefill(self, other):
        """
        Prefill from another KV cache. Shapes must match except batch (other may be batch=1)
        and seq_len (self may be longer).
        """
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with an empty KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                # num_layers, 2(K/V), num_heads, head_dim must match
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                # batch dimension can expand
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                # self must be at least as long
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # lazy init
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)

        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add

        # grow time axis if needed
        if t1 > self.kv_cache.size(4):
            t_needed = (t1 + 1023) & ~1023  # round up to multiple of 1024
            add_shape = list(self.kv_cache.shape)
            add_shape[4] = t_needed - self.kv_cache.size(4)
            additional = torch.empty(add_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape

        # write K/V for this (local) layer
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v

        # return views up to current position
        key_view   = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]

        # advance time when the *last layer of THIS cache* writes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


def norm(x):
    # Purely functional RMSNorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # (B, T, H, D_head) or (B, H, T, D_head) prior to transpose
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    return out.to(x.dtype)


class ExitHead(nn.Module):
    """Exit classification head for early exit."""
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.exit_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        # x: (B, T, n_embd) -> logits: (B, T, vocab)
        return self.exit_head(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_local_idx: int):
        super().__init__()
        # IMPORTANT: this is the *block-local* layer index (0..n_layers_per_block-1)
        self.layer_idx = layer_local_idx

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head    * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache: KVCache | None):
        """
        x: (B, T, C)
        cos_sin: rotary buffers matching the time window
        kv_cache: per-block cache, indexed by block-local layer_idx
        """
        B, T, C = x.size()

        # queries
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        q = norm(q)
        q = q.transpose(1, 2)  # (B, Hq, T, Dh)

        # keys/values
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        k = apply_rotary_emb(k, cos, sin)
        k = norm(k)
        k, v = k.transpose(1, 2), v.transpose(1, 2)  # (B, Hk, T, Dh)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size(2)  # queries in this pass
        Tk = k.size(2)  # total keys available

        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            # training or no prefix: standard causal
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # single-step decode attends full prefix
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # chunked queries with prefix + intra-chunk causality
            prefix_len = Tk - Tq
            # PyTorch: True=MASK, False=KEEP
            attn_mask = torch.ones((Tq, Tk), dtype=torch.bool, device=q.device)
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = False  # keep all prefix positions
            # within-chunk: keep lower triangle (including diagonal)
            attn_mask[:, prefix_len:] = ~torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    """Single transformer layer (attention + MLP). Uses a *block-local* layer index for KV."""
    def __init__(self, config, layer_local_idx: int):
        super().__init__()
        self.layer_local_idx = layer_local_idx
        self.attn = CausalSelfAttention(config, layer_local_idx)
        self.mlp  = MLP(config)

    def forward(self, x, cos_sin, kv_cache: KVCache | None):
        attn_out = self.attn(norm(x), cos_sin, kv_cache)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x


class TransformerBlock(nn.Module):
    """
    A block of N transformer layers with optional per-layer early exit heads.
    Layers inside a block all share the same KVCache object (passed into this block).
    """
    def __init__(self, config, start_layer_idx, n_layers_per_block, is_last_block=False):
        super().__init__()
        self.config = config
        self.start_layer_idx = start_layer_idx  # global layer id of the first layer of this block
        self.n_layers = n_layers_per_block
        self.is_last_block = is_last_block

        # N layers with *local* indices 0..n_layers_per_block-1
        self.layers = nn.ModuleList([Block(config, i) for i in range(n_layers_per_block)])

        # Exit heads: use nn.Identity() sentinel instead of None to keep ModuleList valid
        self.exit_heads = nn.ModuleList()
        if config.use_early_exit:
            for i in range(n_layers_per_block):
                if is_last_block and i == n_layers_per_block - 1:
                    self.exit_heads.append(nn.Identity())
                else:
                    self.exit_heads.append(ExitHead(config.n_embd, config.vocab_size))
        else:
            for _ in range(n_layers_per_block):
                self.exit_heads.append(nn.Identity())

    def forward(self, x, cos_sin, kv_cache: KVCache | None = None, use_early_exit=False, ponder_mask=None):
        """
        x: (B, T, C)
        kv_cache: the *per-block* KV cache shared by layers in this block
        use_early_exit: whether to evaluate exit gates
        ponder_mask: tokens eligible to exit (B, T) or (B, T, 1)
        Returns: x, layer_exit_info (list of (layer_idx_in_block, exit_logits, exited_mask))
        """
        B, T, C = x.size()
        layer_exit_info = []

        if ponder_mask is not None and ponder_mask.dim() == 2:
            ponder_mask = ponder_mask.unsqueeze(-1)
        can_exit = ponder_mask.squeeze(-1).bool() if ponder_mask is not None else torch.ones(B, T, dtype=torch.bool, device=x.device)
        exited_in_block = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        for layer_idx, (layer, exit_head) in enumerate(zip(self.layers, self.exit_heads)):
            # compute the layer
            x_new = layer(x, cos_sin, kv_cache)

            # freeze tokens that already exited in this block
            if use_early_exit and exited_in_block.any():
                x = torch.where(exited_in_block.unsqueeze(-1), x, x_new)
            else:
                x = x_new

            # evaluate gate
            exit_logits = None
            if not isinstance(exit_head, nn.Identity):
                exit_logits = exit_head(norm(x))
                layer_exit_info.append((layer_idx, exit_logits, None))

                if use_early_exit:
                    with torch.no_grad():
                        probs = F.softmax(exit_logits.float(), dim=-1)  # temperature-agnostic gate
                        max_probs = probs.max(dim=-1)[0]                 # (B, T)
                        should_exit = (max_probs >= self.config.exit_threshold) & can_exit & (~exited_in_block)
                        if should_exit.any():
                            exited_in_block = exited_in_block | should_exit
                            layer_exit_info[-1] = (layer_idx, exit_logits, should_exit)

        return x, layer_exit_info


class GPT(nn.Module):
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.config = config

        assert config.n_layer % config.n_layers_per_block == 0, \
            f"n_layer ({config.n_layer}) must be divisible by n_layers_per_block ({config.n_layers_per_block})"
        self.n_blocks = config.n_layer // config.n_layers_per_block

        blocks = []
        for block_idx in range(self.n_blocks):
            start_layer_idx = block_idx * config.n_layers_per_block
            is_last = (block_idx == self.n_blocks - 1)
            blocks.append(TransformerBlock(config, start_layer_idx, config.n_layers_per_block, is_last))

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList(blocks),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # rotary buffers
        self.rotary_seq_len = config.sequence_len * 4
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.last_exit_distribution = None

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights (your original behavior)
        torch.nn.init.zeros_(self.lm_head.weight)

        # zero out projection weights and exit-heads (keep Identities untouched)
        for block in self.transformer.h:
            for layer in block.layers:
                nn.init.zeros_(layer.mlp.c_proj.weight)
                nn.init.zeros_(layer.attn.c_proj.weight)
            for exit_head in block.exit_heads:
                if isinstance(exit_head, ExitHead):
                    nn.init.zeros_(exit_head.exit_head.weight)

        # refresh rotary in case device changed
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out, fan_in = module.weight.size(0), module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.02, matrix_lr=0.02, exit_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd

        matrix_params = []
        exit_params = []
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        for block in self.transformer.h:
            for layer in block.layers:
                matrix_params.extend(layer.parameters())
            for exit_head in block.exit_heads:
                if isinstance(exit_head, ExitHead):
                    exit_params.extend(exit_head.parameters())

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling the LR for the AdamW parameters √1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(params=lm_head_params,    lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params,  lr=embedding_lr   * dmodel_lr_scale),
            dict(params=matrix_params,     lr=matrix_lr      * dmodel_lr_scale),
        ]
        if exit_params:
            param_groups.append(dict(params=exit_params, lr=exit_lr * dmodel_lr_scale))

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_caches=None, loss_reduction='mean',
                return_exit_info=False, use_early_exit_training=None, ponder_mask=None):
        """
        Training/eval forward.
        kv_caches: None or list[KVCache] with length == n_blocks
        """
        B, T = idx.size()
        if use_early_exit_training is None:
            use_early_exit_training = self.config.use_early_exit

        # prepare ponder mask
        if ponder_mask is not None and ponder_mask.dim() == 2:
            ponder_mask = ponder_mask.unsqueeze(-1)

        # rotary window start
        if kv_caches is None:
            T0 = 0
        else:
            assert len(kv_caches) == self.n_blocks
            T0 = kv_caches[0].get_pos()
        assert T0 + T <= self.cos.size(1), "Sequence length grew beyond rotary cache"
        cos_sin = (self.cos[:, T0:T0+T], self.sin[:, T0:T0+T])

        # trunk
        x = norm(self.transformer.wte(idx))
        all_exit_info = []

        for block_idx, block in enumerate(self.transformer.h):
            kv_cache_block = None if kv_caches is None else kv_caches[block_idx]
            x, layer_exit_info = block(
                x, cos_sin, kv_cache=kv_cache_block,
                use_early_exit=use_early_exit_training and targets is not None,
                ponder_mask=ponder_mask
            )
            for layer_in_block, exit_logits, exited_mask in layer_exit_info:
                global_layer_idx = block_idx * self.config.n_layers_per_block + layer_in_block
                all_exit_info.append((block_idx, layer_in_block, global_layer_idx, exit_logits, exited_mask))

        x = norm(x)
        logits = self.lm_head(x).float()

        if targets is not None:
            # main loss
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )

            # expected steps (for stats)
            expected_steps = torch.full((B, T), float(self.config.n_layer), device=idx.device)
            for block_idx, layer_in_block, global_layer, exit_logits, exited_mask in all_exit_info:
                if exited_mask is not None and exited_mask.any():
                    expected_steps = torch.where(
                        exited_mask & (expected_steps == self.config.n_layer),
                        torch.full_like(expected_steps, float(global_layer + 1)),
                        expected_steps
                    )
            if ponder_mask is not None:
                mask = ponder_mask.squeeze(-1).bool() & (targets != -1)
                self.last_expected_steps = ((expected_steps * mask.float()).sum().item() / mask.sum().item()) if mask.any() else self.config.n_layer
            else:
                mask = (targets != -1)
                self.last_expected_steps = ((expected_steps * mask.float()).sum().item() / mask.sum().item()) if mask.any() else self.config.n_layer

            # auxiliary exit loss + (optional) penalty, unchanged structurally
            exit_loss = 0.0
            exit_penalty = 0.0
            if self.config.use_early_exit and all_exit_info:
                total_exit_loss = 0.0
                num_heads = 0
                can_ponder = ponder_mask.squeeze(-1).bool() if ponder_mask is not None else torch.ones(B, T, dtype=torch.bool, device=idx.device)

                for block_idx, layer_in_block, global_layer, exit_logits, exited_mask in all_exit_info:
                    if exit_logits is not None:
                        lx = exit_logits.float()
                        head_loss = F.cross_entropy(
                            lx.view(-1, lx.size(-1)),
                            targets.reshape(-1),
                            ignore_index=-1,
                            reduction=loss_reduction
                        )
                        total_exit_loss += head_loss
                        num_heads += 1

                        if exited_mask is not None and exited_mask.any():
                            mask = exited_mask & can_ponder & (targets != -1)
                            if mask.any():
                                ponder_count = (can_ponder & (targets != -1)).sum().float()
                                if ponder_count > 0:
                                    weight = mask.sum().float() / ponder_count
                                    exit_penalty += (global_layer + 1) * weight * self.config.exit_penalty

                exit_loss = total_exit_loss / num_heads if num_heads > 0 else 0.0

            self.last_exit_loss = exit_loss if isinstance(exit_loss, float) else exit_loss.item()
            self.last_exit_penalty = exit_penalty if isinstance(exit_penalty, float) else exit_penalty.item()

            if return_exit_info:
                # batch exit distribution (only for tokens counted in loss)
                mask = (targets != -1) & (ponder_mask.squeeze(-1).bool() if ponder_mask is not None else torch.ones_like(targets, dtype=torch.bool))
                if mask.any():
                    dist = []
                    for layer_idx in range(self.config.n_layer + 1):
                        if layer_idx == self.config.n_layer:
                            count = ((expected_steps == float(self.config.n_layer)) & mask).sum().item()
                        else:
                            count = ((expected_steps == float(layer_idx + 1)) & mask).sum().item()
                        dist.append(count)
                    self.last_exit_distribution = dist

            return task_loss + exit_loss + exit_penalty
        else:
            # inference/eval forward
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, ponder_mask=None, temperature=1.0, top_k=None, seed=42, use_early_exit=None):
        """
        Decode with per-block KV caches. Exit heads gate inside a block, but we always finish the model
        and sample from lm_head (no sampling from exit heads).
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = torch.Generator(device=device).manual_seed(seed) if temperature > 0 else None

        if use_early_exit is None:
            use_early_exit = self.config.use_early_exit

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        if ponder_mask is not None:
            ponder_mask = torch.tensor([ponder_mask], dtype=torch.float32, device=device)

        # build per-block KV caches
        head_dim = self.config.n_embd // self.config.n_head
        seq_cap = max(len(tokens) + max_tokens, self.config.sequence_len)
        kv_caches = [
            KVCache(
                batch_size=1,
                num_heads=self.config.n_kv_head,
                seq_len=seq_cap,
                head_dim=head_dim,
                num_layers=self.config.n_layers_per_block,
            )
            for _ in range(self.n_blocks)
        ]

        # prefill prompt across all blocks
        _ = self.forward(ids, kv_caches=kv_caches, ponder_mask=ponder_mask)

        exit_counts = [0] * (self.config.n_layer + 1)  # last bucket = "no early exit"

        for _ in range(max_tokens):
            last = ids[:, -1:]
            T0 = kv_caches[0].get_pos()
            cos_sin = (self.cos[:, T0:T0+1], self.sin[:, T0:T0+1])

            x = norm(self.transformer.wte(last))

            allow_gate = use_early_exit and (ponder_mask is None or ponder_mask[:, -1].item() == 1)
            token_mask = (ponder_mask[:, -1:] if ponder_mask is not None
                          else torch.ones_like(last, dtype=torch.float32, device=device))

            earliest_exit = None
            for b_idx, block in enumerate(self.transformer.h):
                x, info = block(
                    x, cos_sin,
                    kv_cache=kv_caches[b_idx],
                    use_early_exit=allow_gate,
                    ponder_mask=token_mask
                )
                if earliest_exit is None:
                    for layer_in_block, _, exited_mask in info:
                        if exited_mask is not None and exited_mask[0, 0].item():
                            earliest_exit = b_idx * self.config.n_layers_per_block + layer_in_block
                            break

            x = norm(x)
            logits = self.lm_head(x)[:, -1, :]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat([ids, next_id], dim=1)

            if ponder_mask is not None:
                ponder_mask = torch.cat(
                    [ponder_mask, torch.ones(1, 1, dtype=torch.float32, device=device)],
                    dim=1
                )

            if earliest_exit is None:
                exit_counts[-1] += 1
            else:
                exit_counts[earliest_exit] += 1

            total = sum(exit_counts)
            avg_layer = (sum(i * c for i, c in enumerate(exit_counts)) / total) if total else float(self.config.n_layer)
            yield next_id.item(), avg_layer

        self.last_exit_distribution = exit_counts
