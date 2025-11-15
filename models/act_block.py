import torch
import torch.nn as nn


class AdaptiveBlock(nn.Module):
    """
    Adaptive computation block containing N transformer layers.
    Performs recursion over all N layers together, sharing K,V across recursion steps.
    """
    def __init__(self, config, start_layer_idx, n_layers_per_block):
        super().__init__()

        
        from models.transformer import HaltingUnit, Block
        self.config = config
        self.start_layer_idx = start_layer_idx
        self.n_layers = n_layers_per_block
        
        # Create N transformer layers
        self.layers = nn.ModuleList([
            Block(config, start_layer_idx + i) 
            for i in range(n_layers_per_block)
        ])
        
        # Halting unit for adaptive computation
        self.halting_unit = HaltingUnit(config.n_embd)
    
    def forward_once(self, x, cos_sin, kv_cache, shared_kvs=None):
        """
        single forward pass through all layers in this block.

        Args:
            x: input (B, T, C)
            cos_sin: rotary embeddings
            shared_kvs: List of (k, v) pairs for each layer (or None for first recursion)
            input_pos: Position indices for RoPE (B, T)
        
        Returns:
            x: output after all layers
            new_kvs: List of K,V pairs from each layer (or None if using shared_kvs)
        """
        # if it is first step -> shared_kvs will be None and new_kvs will be populated
        new_kvs = []
        for i, layer in enumerate(self.layers):
            shared_kv = shared_kvs[i] if shared_kvs is not None else None
            x, kv = layer(x, cos_sin, kv_cache, shared_kv=shared_kv)
            new_kvs.append(kv)
        
        return x, new_kvs if shared_kvs is None else None
    
    def forward(self, x, cos_sin, kv_cache=None, ponder_mask=None):
        B, T, C = x.size()
        device = x.device
        config = self.config
        if ponder_mask is None:
            ponder_mask = torch.ones(B, T, 1, device=device)
            ponder_mask = ponder_mask.to(device)

        assert ponder_mask.shape[:2] == (B, T), f"{ponder_mask.shape} vs {(B,T)}"

        forced_halt = (ponder_mask == 0)

        accumulated_output = torch.zeros_like(x)
        accumulated_probs  = torch.zeros(B, T, 1, device=device)
        num_updates        = torch.zeros(B, T, 1, device=device)  
        remainder_at_halt  = torch.zeros(B, T, 1, device=device)  
        halted             = torch.zeros(B, T, dtype=torch.bool, device=device)

        cached_kvs = None
        x_current = x

        for step in range(config.max_pondering_steps):
            # count this update for tokens that are not yet halted
            num_updates = num_updates + (~halted).unsqueeze(-1).float()

            if step == 0:
                x_step, kvs = self.forward_once(x_current, cos_sin, kv_cache, shared_kvs=None)  #no shared_kvs
                cached_kvs = kvs
            else:
                x_step, _   = self.forward_once(x_current, cos_sin, kv_cache, shared_kvs=cached_kvs)

            halt_prob = self.halting_unit(x_step)  # (B,T,1)

            if step == 0:
                halt_prob = torch.where(forced_halt, torch.ones_like(halt_prob), halt_prob)

            is_last_step = (step == config.max_pondering_steps - 1)
            should_halt  = (accumulated_probs + halt_prob >= config.act_threshold) | is_last_step

            # mass assigned this step (should_halt also contains already halted -> will be assigned with remaining_mass which is <1.0 - accumulated_probs>)
            remaining_mass = (1.0 - accumulated_probs)
            step_weight = torch.where(should_halt, remaining_mass, halt_prob)  # p_t

            # capture R(t) for tokens halted at this state
            new_halts = should_halt & (~halted.unsqueeze(-1))
            remainder_at_halt = torch.where(new_halts, remaining_mass, remainder_at_halt)

            # accumulate output and probabilities
            accumulated_output = accumulated_output + step_weight * x_step
            accumulated_probs  = accumulated_probs  + step_weight   

            halted = halted | should_halt.squeeze(-1)
            x_current = x_step

            if halted.all():
                break

        # R(t): differentiable remainder
        N_detached = num_updates.detach()
        act_ponder = remainder_at_halt + N_detached  # shape (B,T,1)

        return accumulated_output, act_ponder, N_detached