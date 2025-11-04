import torch
from torch.utils.data import DataLoader

from tokenizer import TinyTokenizer
from dataset.synthetic_tasks import AddMul, Difficulty, make_collate_fn, Tokenizer
from models.transformer import GPT, TransformerBlockConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    tokenizer = Tokenizer()
    max_seq_len = 128

    ds = AddMul(num_samples=50_000, max_operands=6, max_digits=6, seed=0)
    collate_fn = make_collate_fn(tokenizer, max_seq_len)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                    collate_fn=collate_fn, drop_last=True)

    cfg = TransformerBlockConfig(
        sequence_len=max_seq_len,
        vocab_size=len(tokenizer),
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        n_layer=2,
        use_adaptive_computation=True,
        n_layers_per_block=2,
        max_pondering_steps=5,
        act_threshold=0.99,
        halting_penalty=0.01,
    )
    model = GPT(cfg).to(device)
    model.init_weights()

    opt = model.setup_optimizers(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
    )

    steps = 2000
    log_every = 50
    model.train()
    for step, batch in enumerate(dl, start=1):
        if step > steps:
            break
        idx = batch["idx"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        ponder_mask = batch["ponder_mask"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(idx, targets=targets, kv_cache=None,
                        loss_reduction="mean", ponder_mask=ponder_mask)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % log_every == 0:
                pc = getattr(model, "last_ponder_cost", float("nan"))
                es = getattr(model, "last_expected_steps", float("nan"))
                ap = getattr(model, "last_act_penalty", float("nan"))
                print(f"step {step:5d}  loss={loss.item():.4f}  ponder_cost={pc:.3f}  "
                    f"exp_steps={es:.3f}  act_penalty={ap:.4f}")


    model.eval()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_seq, output_seq = ds[0]
            print("Problem:", "".join(input_seq), "Answer:", "".join(output_seq))
            prompt_ids = tokenizer.encode(input_seq)  # includes '='
            outs = []
            for tok in model.generate(tokens=prompt_ids, max_tokens=40, temperature=0.0):
                outs.append(tok)
                if tok == tokenizer.eos_id:
                    break
            print("Model output:", "".join(tokenizer.decode(outs)))

if __name__ == "__main__":
    main()
