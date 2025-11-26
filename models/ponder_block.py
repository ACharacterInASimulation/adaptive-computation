import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePonderBlock(nn.Module):
    """
    PonderNet variant of the AdaptiveBlock.
    - Same architecture (N transformer layers + HaltingUnit)
    - Each block defines a halting distribution p_t over pondering steps.
    - Computes expectation over step outputs and KL(p || geometric_prior).
    """

    def __init__(self, config, start_layer_idx, n_layers_per_block):
        super().__init__()
        from models.transformer import HaltingUnit, Block

        self.config = config
        self.start_layer_idx = start_layer_idx
        self.n_layers = n_layers_per_block

        # N transformer layers per block
        self.layers = nn.ModuleList([
            Block(config, start_layer_idx + i)
            for i in range(n_layers_per_block)
        ])

        # Halting head (same as in ACT)
        self.halting_unit = HaltingUnit(config.n_embd)


    def forward_once(self, x, cos_sin, kv_cache, shared_kvs=None):
        """
        Single forward pass through all layers in this block.
        Handles shared-KV reuse exactly like the ACT version.
        """
        new_kvs = []
        for i, layer in enumerate(self.layers):
            shared_kv = shared_kvs[i] if shared_kvs is not None else None
            x, kv = layer(x, cos_sin, kv_cache, shared_kv=shared_kv)
            new_kvs.append(kv)
        return x, (new_kvs if shared_kvs is None else None)

    # --------------------------------------------------------
    # PonderNet-specific halting logic
    # --------------------------------------------------------
    def forward(self, x, cos_sin, kv_cache=None, ponder_mask=None):
        """
        Forward pass with PonderNet halting.
        Returns:
            y_expect      : [B,T,C]  expectation over pondering steps
            kl_term       : scalar   KL(p || geometric prior)
            expected_steps: [B,T,1]  per-token expected number of steps
        """
        B, T, C = x.size()
        device = x.device
        cfg = self.config

        if ponder_mask is None:
            ponder_mask = torch.ones(B, T, 1, device=device)
        else:
            assert ponder_mask.shape[:2] == (B, T), f"{ponder_mask.shape} vs {(B,T)}"

        forced_halt = (ponder_mask == 0)  # pad or invalid tokens

        # --- Initialize accumulators ---
        y_expect  = torch.zeros_like(x)
        survival  = torch.ones(B, T, 1, device=device)  # S_{t-1}
        cdf       = torch.zeros(B, T, 1, device=device)
        kl_sum    = torch.zeros((), device=device)
        exp_steps = torch.zeros(B, T, 1, device=device)

        cached_kvs = None
        x_curr = x

        # Geometric prior parameter λ
        λ = max(min(cfg.geom_lambda_prior, 1.0 - 1e-6), 1e-6)

        # ----------------------------------------------------
        # Main pondering loop
        # ----------------------------------------------------
        for t in range(1, cfg.max_pondering_steps + 1):
            # Run one ponder step through N layers
            if t == 1:
                y_t, new_kvs = self.forward_once(x_curr, cos_sin, kv_cache, shared_kvs=None)
                cached_kvs = new_kvs
            else:
                y_t, _ = self.forward_once(x_curr, cos_sin, kv_cache, shared_kvs=cached_kvs)

            # Hazard (halting probability) ∈ (0,1)
            h_t = torch.sigmoid(self.halting_unit(y_t))

            # Force immediate halt for masked tokens (pads)
            if t == 1:
                h_t = torch.where(forced_halt, torch.ones_like(h_t), h_t)

            # Halting pmf: p_t = h_t * survival_{t-1}
            p_t = h_t * survival
            p_t = torch.clamp(p_t, min=1e-8)  # numerical safety

            # Update accumulators
            y_expect  = y_expect  + p_t * y_t
            cdf       = cdf       + p_t
            exp_steps = exp_steps + t * p_t

            # Geometric prior q_t = λ (1-λ)^{t-1}
            log_q_t = math.log(λ) + (t - 1) * math.log(1 - λ)
            kl_sum  = kl_sum + torch.sum(p_t * (torch.log(p_t) - log_q_t))

            # Update survival for next step
            survival = survival * (1.0 - h_t)

            # Optional early stop when most mass assigned
            if cdf.mean().item() >= getattr(cfg, "cdf_early_stop", 0.999):
                residual = torch.clamp(1.0 - cdf, min=0.0)
                if (residual > 0).any():
                    y_expect  = y_expect  + residual * y_t
                    exp_steps = exp_steps + t * residual
                    kl_sum    = kl_sum + torch.sum(
                        residual * (torch.log(torch.clamp(residual, 1e-8)) - log_q_t)
                    )
                break

            x_curr = y_t

        # If truncated: push leftover mass to last step
        if cdf.mean().item() < 0.999 and t == cfg.max_pondering_steps:
            residual = torch.clamp(1.0 - cdf, min=0.0)
            if (residual > 0).any():
                y_expect  = y_expect  + residual * y_t
                exp_steps = exp_steps + t * residual
                kl_sum    = kl_sum + torch.sum(
                    residual * (torch.log(torch.clamp(residual, 1e-8)) - math.log(λ) - (t - 1) * math.log(1 - λ))
                )

        # Mean KL over tokens
        kl_term = kl_sum / (B * T)
        expected_steps = exp_steps.detach()

        return y_expect, kl_term, expected_steps
