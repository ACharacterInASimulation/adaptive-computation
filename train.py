import torch
from torch.utils.data import DataLoader
import wandb
import os
import yaml

from dataset.synthetic_tasks import AddMul, Tokenizer, generate_eval
from dataset.synthetic_tasks import make_collate_fn as make_synthetic_collate_fn
from dataset.babylm import BabyLMDataset, HuggingFaceTokenizer, evaluate_babylm
from dataset.babylm import make_collate_fn as make_babylm_collate_fn
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
    
    # Use custom name if provided, otherwise generate from config
    if wandb_cfg.get("name"):
        name = wandb_cfg["name"]
    else:
        # Extract important config info
        dataset = config['data']['dataset_name']
        n_embd = model_cfg.n_embd
        n_layer = model_cfg.n_layer
        n_head = model_cfg.n_head
        
        # Check for shared KV (if n_kv_head < n_head, it's grouped query attention)
        if model_cfg.n_kv_head < model_cfg.n_head:
            kv_type = f"GQA{model_cfg.n_kv_head}"
        else:
            kv_type = "MHA"
        
        # Check if KV cache is shared across ACT blocks
        share_kv = getattr(model_cfg, 'share_kv', False)
        if share_kv:
            share_kv_layers = getattr(model_cfg, 'share_kv_n_layers', n_layer)
            kv_suffix = f"-SharedKV{share_kv_layers}"
        else:
            kv_suffix = ""
        
        # Check if using recursive model
        recursive = getattr(model_cfg, 'recursive', False)
        if recursive:
            recursion_depth = getattr(model_cfg, 'recursion_depth', 5)
            recursive_suffix = f"-Rec{recursion_depth}"
        else:
            recursive_suffix = ""
        
        if model_cfg.use_adaptive_computation:
            max_steps = model_cfg.max_pondering_steps
            n_blocks = model_cfg.n_layers_per_block
            threshold = model_cfg.act_threshold
            penalty = model_cfg.halting_penalty
            name = f"{dataset}-d{n_embd}-l{n_layer}-h{n_head}-{kv_type}{kv_suffix}{recursive_suffix}-ACT{max_steps}x{n_blocks}-th{threshold}-pen{penalty}"
        elif model_cfg.use_pondernet:
            max_steps = model_cfg.max_pondering_steps
            n_blocks = model_cfg.n_layers_per_block
            geom_lambda = model_cfg.geom_lambda_prior
            kl_beta = model_cfg.kl_weight_beta
            name = f"{dataset}-d{n_embd}-l{n_layer}-h{n_head}-{kv_type}{kv_suffix}{recursive_suffix}-Ponder{max_steps}x{n_blocks}-λ{geom_lambda}-β{kl_beta}"
        else:
            name = f"{dataset}-d{n_embd}-l{n_layer}-h{n_head}-{kv_type}{kv_suffix}{recursive_suffix}-baseline"

    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=name,
        tags=wandb_cfg["tags"],
        notes=wandb_cfg["notes"],
        config=config,
    )
    return run


def main(config_file="configs/babylm_act.yaml"):
    '''Main training'''

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
        # Load or train BabyLM tokenizer
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
        shuffle=DATA_CFG.get("shuffle", False),
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
        share_kv=MODEL_CFG.get("share_kv", False),
        share_kv_n_layers=MODEL_CFG.get("share_kv_n_layers", 0),
        recursive=MODEL_CFG.get("recursive", False),
        recursion_depth=MODEL_CFG.get("recursion_depth", 5),
        use_adaptive_computation=MODEL_CFG.get("use_adaptive_computation", False),
        n_layers_per_block=MODEL_CFG["n_layers_per_block"],
        max_pondering_steps=MODEL_CFG["max_pondering_steps"],
        act_threshold=MODEL_CFG.get("act_threshold", 0.99),
        halting_penalty=MODEL_CFG.get("halting_penalty", 0.01),
        # PonderNet config
        use_pondernet=MODEL_CFG.get("use_pondernet", False),
        geom_lambda_prior=MODEL_CFG.get("geom_lambda_prior", 0.5),
        kl_weight_beta=MODEL_CFG.get("kl_weight_beta", 0.05),
        cdf_early_stop=MODEL_CFG.get("cdf_early_stop", 0.999),
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
        if batch["ponder_mask"] is not None:
            ponder_mask = batch["ponder_mask"].to(device, non_blocking=True)
        else:
            ponder_mask = None

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
        elif cfg.use_pondernet:
            # PonderNet: loss already includes KL term from forward_pondernet
            actual_loss, _ = model(idx, targets=targets, kv_cache=None,
                        loss_reduction="mean", ponder_mask=ponder_mask)
            task_loss = actual_loss  # For logging purposes
            penalty_scale = cfg.kl_weight_beta  # Log the KL weight
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
                elif cfg.use_pondernet:
                    ponder_kl = getattr(model, "last_ponder_kl", float("nan"))
                    log_dict.update({
                        "ponder/expected_steps": es,
                        "ponder/kl_loss": ponder_kl,
                        "ponder/kl_weight": cfg.kl_weight_beta,
                    })
                wandb.log(log_dict, step=step)
        
        # Evaluation
        if step % TRAIN_CFG["eval_every"] == 0:
            model.eval()
            if dataset_name == "AddMul":
                metrics = generate_eval(model, val_ds, tokenizer, n_samples=TRAIN_CFG["eval_n_samples"])

                if config["wandb"]["enabled"]:
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
                
                print(f"\nEvaluation at step {step}:")
                for k, v in metrics.items():
                    print(f"  {k}: {v}" if isinstance(v, float) else f"  {k}: {v}")


            elif dataset_name == "BabyLM" and val_dl is not None:
                metrics = evaluate_babylm(model, val_dl, device, cfg)
                
                if config["wandb"]["enabled"]:
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
                
                print(f"val_loss: {metrics['val_loss']:.4f}")
                print(f"val_perplexity: {metrics['val_perplexity']:.4f}")
                if "val_expected_steps" in metrics:
                    print(f"val_expected_steps: {metrics['val_expected_steps']}")


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
