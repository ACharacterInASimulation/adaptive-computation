import torch
from torch.utils.data import DataLoader
import wandb
import os
import yaml

from dataset.synthetic_tasks import AddMul, make_collate_fn, Tokenizer, generate_eval
from models.transformer import GPT, TransformerBlockConfig


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config, override_config):
    merged = base_config.copy()
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def init_wandb(model_cfg, config):
    wandb_cfg = config["wandb"]
    if not wandb_cfg["enabled"]:
        return None
    
    name = wandb_cfg["name"]
    if name is None:
        name = f"gpt-{model_cfg.n_embd}x{model_cfg.n_layer}x{model_cfg.max_pondering_steps}-act" if model_cfg.use_adaptive_computation else f"gpt-{model_cfg.n_embd}x{model_cfg.n_layer}-baseline"

    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=name,
        tags=wandb_cfg["tags"],
        notes=wandb_cfg["notes"],
        config=config,
    )
    return run


def main(config_file="configs/default.yaml"):
    '''Main training function.'''

    print(f"Loading config from: {config_file}")
    config = load_config(config_file)
    
 
    if config_file != "configs/default.yaml":
        default_config = load_config("configs/default.yaml")
        config = merge_configs(default_config, config)
    
    MODEL_CFG = config["model"]
    DATA_CFG = config["data"]
    OPT_CFG = config["optimizer"]
    TRAIN_CFG = config["training"]
    CKPT_CFG = config["checkpoint"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(TRAIN_CFG["seed"])
    torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}")

    tokenizer = Tokenizer()

    # Dataset setup
    ds = AddMul(
        num_samples=DATA_CFG["num_samples"],
        max_operands=DATA_CFG["max_operands"],
        max_digits=DATA_CFG["max_digits"],
        seed=DATA_CFG["seed"],
        operations=DATA_CFG["operations"],
    )
    collate_fn = make_collate_fn(tokenizer, MODEL_CFG["sequence_len"])
    
    dl = DataLoader(
        ds,
        batch_size=TRAIN_CFG["batch_size"],
        shuffle=False,
        num_workers=TRAIN_CFG["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )

    #model configuration
    cfg = TransformerBlockConfig(
        sequence_len=MODEL_CFG["sequence_len"],
        vocab_size=len(tokenizer),
        n_head=MODEL_CFG["n_head"],
        n_kv_head=MODEL_CFG["n_kv_head"],
        n_embd=MODEL_CFG["n_embd"],
        n_layer=MODEL_CFG["n_layer"],
        use_adaptive_computation=MODEL_CFG["use_adaptive_computation"],
        n_layers_per_block=MODEL_CFG["n_layers_per_block"],
        max_pondering_steps=MODEL_CFG["max_pondering_steps"],
        act_threshold=MODEL_CFG["act_threshold"],
        halting_penalty=MODEL_CFG["halting_penalty"],
    )

    model = GPT(cfg).to(device)
    model.init_weights()
    model = model.float()
    
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")


    opt = model.setup_optimizers(
        unembedding_lr=OPT_CFG["unembedding_lr"],
        embedding_lr=OPT_CFG["embedding_lr"],
        matrix_lr=OPT_CFG["matrix_lr"],
        halting_lr=OPT_CFG["halting_lr"],
        weight_decay=OPT_CFG["weight_decay"],
    )
    

    all_config = {
        "model": MODEL_CFG,
        "data": DATA_CFG,
        "optimizer": OPT_CFG,
        "training": TRAIN_CFG,
        "wandb": config["wandb"],
        "checkpoint": CKPT_CFG,
    }
    
    # Initialize wandb
    init_wandb(cfg, all_config)
    

    os.makedirs(CKPT_CFG["save_dir"], exist_ok=True)
    
    model.train()
    print(f"Starting training for {TRAIN_CFG['steps']} steps...")
    
    for step, batch in enumerate(dl, start=1):
        if step > TRAIN_CFG["steps"]:
            break
            
        idx = batch["idx"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        ponder_mask = batch["ponder_mask"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        if cfg.use_adaptive_computation:
            task_loss, act_penalty = model(idx, targets=targets, kv_cache=None,
                        loss_reduction="mean", ponder_mask=ponder_mask)

            #warmup: reduce penalty initially
            if step <= TRAIN_CFG["warmup_steps"] and TRAIN_CFG["warmup_steps"] > 0:
                penalty_scale = min(1.0, step / TRAIN_CFG["warmup_steps"])
            else:
                penalty_scale = 1.0

            actual_loss = task_loss + penalty_scale * act_penalty
        else:
            task_loss, _ = model(idx, targets=targets, kv_cache=None,
                        loss_reduction="mean", ponder_mask=ponder_mask)
            actual_loss = task_loss
            penalty_scale = 0.0
            
        actual_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["gradient_clip"])
        opt.step()


        if step % TRAIN_CFG["log_every"] == 0:
            es = getattr(model, "last_expected_steps", float("nan"))
            
            print(f"step {step:8d}  loss={task_loss.item():.4f}  "
                  f"exp_steps={es:.3f}  penalty×{penalty_scale:.5f}")

            # Log to wandb
            if config["wandb"]["enabled"]:
                log_dict = {
                    "train/loss": task_loss.item(),
                    "train/total_loss": actual_loss.item(),
                }
                if cfg.use_adaptive_computation:
                    log_dict.update({
                        "act/expected_steps": es,
                        "act/penalty_scale": penalty_scale,
                    })
                wandb.log(log_dict, step=step)
        
        # Evaluation
        if step % TRAIN_CFG["eval_every"] == 0:
            model.eval()
            metrics = generate_eval(model, ds, tokenizer, n_samples=TRAIN_CFG["eval_n_samples"])
            

            if config["wandb"]["enabled"]:
                wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            
 
            print(f"\nEvaluation at step {step}:")
            for k, v in metrics.items():
                print(f"  {k}: {v}" if isinstance(v, float) else f"  {k}: {v}")
            print()
            
            model.train()
    

    if CKPT_CFG["save_final"]:
        checkpoint_path = os.path.join(CKPT_CFG["save_dir"], CKPT_CFG["final_name"])
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'config': cfg,
            'step': step,
        }, checkpoint_path)
        print(f"Saved final model to {checkpoint_path}")

    if config["wandb"]["enabled"]:
        wandb.finish()
    print("Training completed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GPT with Adaptive Computation Time")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to YAML config file")
    args = parser.parse_args()
    
    main(args.config)
