import torch
from torch.utils.data import DataLoader
import wandb
import os
import yaml

from dataset.synthetic_tasks import AddMul, Tokenizer, generate_eval
from dataset.synthetic_tasks import make_collate_fn as make_synthetic_collate_fn
from dataset.babylm import BabyLMDataset, HuggingFaceTokenizer, evaluate_babylm
from dataset.babylm import make_collate_fn as make_babylm_collate_fn
from models.transformer_earlyexit import GPT, TransformerBlockConfig


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
    
    # Use custom name if provided, otherwise generate from config
    if wandb_cfg.get("name"):
        name = wandb_cfg["name"]
    else:
        # Extract important config info
        dataset = config['data']['dataset_name']
        n_embd = model_cfg.n_embd
        n_layer = model_cfg.n_layer
        n_head = model_cfg.n_head
        
        # Check for grouped query attention
        if model_cfg.n_kv_head < model_cfg.n_head:
            kv_type = f"GQA{model_cfg.n_kv_head}"
        else:
            kv_type = "MHA"
        
        if model_cfg.use_early_exit:
            n_blocks = model_cfg.n_layers_per_block
            threshold = model_cfg.exit_threshold
            penalty = model_cfg.exit_penalty
            name = f"{dataset}-d{n_embd}-l{n_layer}-h{n_head}-{kv_type}-EarlyExit{n_blocks}blk-th{threshold}-pen{penalty}"
        else:
            name = f"{dataset}-d{n_embd}-l{n_layer}-h{n_head}-{kv_type}-baseline"

    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=name,
        tags=wandb_cfg["tags"],
        notes=wandb_cfg["notes"],
        config=config,
    )
    return run


def evaluate_babylm_early_exit(model, val_dataloader, device, cfg):
    """Evaluate BabyLM with early exit tracking."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            idx = batch["idx"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            
            # Forward pass
            loss = model(idx, targets=targets, kv_cache=None,
                        loss_reduction="sum", return_exit_info=True,
                        use_early_exit_training=False)
            
            batch_tokens = targets.numel()
            total_loss += loss.item()
            total_tokens += batch_tokens
            num_batches += 1
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    metrics = {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
    }
    
    return metrics


def main(config_file="configs/earlyexit_babylm.yaml"):
    '''Main training for early exit transformer'''

    print(f"Loading config from: {config_file}")
    config = load_config(config_file)
    
    MODEL_CFG = config["model"]
    DATA_CFG = config["data"]
    OPT_CFG = config["optimizer"]
    TRAIN_CFG = config["training"]
    CKPT_CFG = config["checkpoint"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(TRAIN_CFG["seed"])
    torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}")

    dataset_name = DATA_CFG.get("dataset_name", "AddMul")
    print(f"Using dataset: {dataset_name}")

    # Dataset and tokenizer setup based on dataset_name
    if dataset_name == "BabyLM":
        # Load BabyLM tokenizer
        tokenizer_path = DATA_CFG.get("tokenizer_path", "checkpoints")
        
        if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
            print(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_path)
        else:
            print(f"Tokenizer not found at {tokenizer_path}. Please train a tokenizer first.")
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        # Create BabyLM dataset
        ds = BabyLMDataset(
            data_dir=DATA_CFG["data_dir"],
            tokenizer=tokenizer,
            seq_len=MODEL_CFG["sequence_len"],
            split="train",
        )
        collate_fn = make_babylm_collate_fn(tokenizer, MODEL_CFG["sequence_len"])
        
    elif dataset_name == "AddMul":
        # Synthetic task dataset
        tokenizer = Tokenizer()
        ds = AddMul(
            num_samples=DATA_CFG["num_samples"],
            max_operands=DATA_CFG["max_operands"],
            max_digits=DATA_CFG["max_digits"],
            seed=DATA_CFG["seed"],
            operations=DATA_CFG["operations"],
        )

        val_ds = AddMul(
            num_samples=TRAIN_CFG["eval_n_samples"],
            max_operands=DATA_CFG["max_operands"],
            max_digits=DATA_CFG["max_digits"],
            seed=DATA_CFG["eval_seed"],
            operations=DATA_CFG["operations"],
        )
        collate_fn = make_synthetic_collate_fn(tokenizer, MODEL_CFG["sequence_len"])
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Must be 'BabyLM' or 'AddMul'")

    print(f"Dataset size: {len(ds)}")
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else len(tokenizer)}")

    dl = DataLoader(
        ds,
        batch_size=TRAIN_CFG["batch_size"],
        shuffle=DATA_CFG.get("shuffle", True),  # Enable shuffling for multi-epoch
        num_workers=TRAIN_CFG["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_dl = None
    if dataset_name == "BabyLM":
        val_ds = BabyLMDataset(
            data_dir=DATA_CFG["val_data_dir"],
            tokenizer=tokenizer,
            seq_len=MODEL_CFG["sequence_len"],
            split="val",
            max_samples=TRAIN_CFG.get("eval_n_samples", None),
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=TRAIN_CFG["batch_size"],
            shuffle=False,
            num_workers=TRAIN_CFG["num_workers"],
            collate_fn=collate_fn,
            drop_last=False,
        )

    # Model configuration
    vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else len(tokenizer)
    cfg = TransformerBlockConfig(
        sequence_len=MODEL_CFG["sequence_len"],
        vocab_size=vocab_size,
        n_head=MODEL_CFG["n_head"],
        n_kv_head=MODEL_CFG["n_kv_head"],
        n_embd=MODEL_CFG["n_embd"],
        n_layer=MODEL_CFG["n_layer"],
        n_layers_per_block=MODEL_CFG["n_layers_per_block"],
        use_early_exit=MODEL_CFG["use_early_exit"],
        exit_threshold=MODEL_CFG["exit_threshold"],
        exit_penalty=MODEL_CFG["exit_penalty"],
    )

    model = GPT(cfg).to(device)
    model.init_weights()
    model = model.float()
    
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    opt = model.setup_optimizers(
        unembedding_lr=OPT_CFG["unembedding_lr"],
        embedding_lr=OPT_CFG["embedding_lr"],
        matrix_lr=OPT_CFG["matrix_lr"],
        exit_lr=OPT_CFG["exit_lr"],
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
    
    # Calculate expected epochs
    steps_per_epoch = len(dl)
    expected_epochs = (TRAIN_CFG['steps'] + steps_per_epoch - 1) // steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Expected epochs: {expected_epochs}")
    
    # Multi-epoch training loop
    global_step = 0
    for epoch in range(expected_epochs):
        print(f"\n=== Epoch {epoch + 1}/{expected_epochs} ===")
        
        for batch_idx, batch in enumerate(dl):
            global_step += 1
            
            if global_step > TRAIN_CFG["steps"]:
                print(f"Reached target steps ({TRAIN_CFG['steps']}), stopping training")
                break
                
            idx = batch["idx"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # Forward pass with early exit
            total_loss = model(idx, targets=targets, kv_cache=None,
                              loss_reduction="mean", return_exit_info=True,
                              use_early_exit_training=cfg.use_early_exit)
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["gradient_clip"])
            opt.step()

            if global_step % TRAIN_CFG["log_every"] == 0:
                exit_loss = getattr(model, "last_exit_loss", 0.0)
                exit_penalty = getattr(model, "last_exit_penalty", 0.0)
                expected_steps = getattr(model, "last_expected_steps", float('nan'))
                
                print(f"step {global_step:8d} (epoch {epoch+1}, batch {batch_idx+1}/{steps_per_epoch})  "
                      f"loss={total_loss.item():.4f}  exit_loss={exit_loss:.4f}  "
                      f"exit_penalty={exit_penalty:.4f}  expected_steps={expected_steps:.2f}")

                # Log to wandb
                if config["wandb"]["enabled"]:
                    log_dict = {
                        "train/total_loss": total_loss.item(),
                        "train/exit_loss": exit_loss,
                        "train/exit_penalty": exit_penalty,
                        "train/expected_steps": expected_steps,
                        "train/epoch": epoch + 1,
                    }
                    
                    # Log exit distribution if available
                    if hasattr(model, "last_exit_distribution") and model.last_exit_distribution is not None:
                        for layer_idx, count in enumerate(model.last_exit_distribution):
                            if layer_idx == cfg.n_layer:
                                log_dict[f"train/exit_dist/final_layer"] = count
                            else:
                                log_dict[f"train/exit_dist/layer_{layer_idx}"] = count
                    
                    wandb.log(log_dict, step=global_step)
            
            # Evaluation
            if global_step % TRAIN_CFG["eval_every"] == 0:
                model.eval()
                if dataset_name == "AddMul":
                    metrics = generate_eval(model, val_ds, tokenizer, n_samples=TRAIN_CFG["eval_n_samples"])

                    if config["wandb"]["enabled"]:
                        wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)
                    
                    print(f"\nEvaluation at step {global_step}:")
                    for k, v in metrics.items():
                        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
                    
                    # Log exit distribution from generation
                    if hasattr(model, "last_exit_distribution") and model.last_exit_distribution is not None:
                        print("\nExit distribution during generation:")
                        for layer_idx, count in enumerate(model.last_exit_distribution):
                            if count > 0:
                                if layer_idx == cfg.n_layer:
                                    print(f"  Final layer: {count}")
                                else:
                                    print(f"  Layer {layer_idx}: {count}")

                elif dataset_name == "BabyLM" and val_dl is not None:
                    metrics = evaluate_babylm_early_exit(model, val_dl, device, cfg)
                    
                    if config["wandb"]["enabled"]:
                        wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)
                    
                    print(f"\nEvaluation at step {global_step}:")
                    print(f"val_loss: {metrics['val_loss']:.4f}")
                    print(f"val_perplexity: {metrics['val_perplexity']:.4f}")

                model.train()
            
            # Periodic checkpointing
            if CKPT_CFG.get("save_every", 0) > 0 and global_step % CKPT_CFG["save_every"] == 0:
                checkpoint_path = os.path.join(CKPT_CFG["save_dir"], f"checkpoint_step_{global_step}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'config': cfg,
                    'step': global_step,
                    'epoch': epoch,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # Break outer loop if we've reached target steps
        if global_step > TRAIN_CFG["steps"]:
            break
    
    if CKPT_CFG["save_final"]:
        checkpoint_path = os.path.join(CKPT_CFG["save_dir"], CKPT_CFG["final_name"])
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'config': cfg,
            'step': global_step,
            'epoch': epoch,
        }, checkpoint_path)
        print(f"Saved final model to {checkpoint_path}")

    if config["wandb"]["enabled"]:
        wandb.finish()
    print(f"Training completed after {global_step} steps across {epoch + 1} epochs")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GPT with Early Exit")
    parser.add_argument("--config", type=str, default="configs/earlyexit_default.yaml",
                       help="Path to YAML config file")
    args = parser.parse_args()
    
    main(args.config)