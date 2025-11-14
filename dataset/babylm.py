'''HF_Tokenizer taken from nanochat'''

from pathlib import Path
from typing import List, Optional
import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, decoders, pre_tokenizers, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = []
TRAIN_FILES = ["gutenberg.train"]
VAL_FILES = ["gutenberg.dev"]

class HuggingFaceTokenizer:
    def __init__(self, tokenizer = None):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        tokenizer = Tokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_files(cls, files, vocab_size):
        # Initialize a HuggingFace Tokenizer (BPE model)
        tokenizer = Tokenizer(BPE(
            byte_fallback=True,
            unk_token=None,
            fuse_unk=False,
        ))

        # Pre-tokenizer: GPT-style regex + byte-level
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])

        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.normalizer = None
        tokenizer.post_processor = None

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train(files, trainer)
        print(f"training tokenizer complete. Vocabulary size: {tokenizer.get_vocab_size()}")
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        return self.encode_special("<|bos|>")

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")



class BabyLMDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: HuggingFaceTokenizer,
        seq_len: int = 512,
        split: str = "train",  
        max_samples: Optional[int] = None,  
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        
        files_to_use = TRAIN_FILES if split == "train" else VAL_FILES
        self.tokens = self._load_and_tokenize(files_to_use)
        self.chunks = self._create_chunks()
        

        if max_samples is not None and max_samples < len(self.chunks):
            self.chunks = self.chunks[:max_samples]
            print(f"[{split}] Limited to {max_samples} samples")
        
        print(f"[{split}] total tokens : {len(self.tokens):,}")
        print(f"[{split}] number of chunks : {len(self.chunks):,}")
        print(f"[{split}] sequence length : {seq_len}")
    
    def _load_and_tokenize(self, files: List[str]) -> List[int]:
        all_tokens = []
        for filename in files:
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                print(f"{filepath} not found")
                continue
            print(f"Loading {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            print(f"added {len(tokens)} tokens from {filename}")

        return all_tokens
    
    def _create_chunks(self) -> List[List[int]]:
        chunks = []
        total_tokens = len(self.tokens)
        for i in range(0, total_tokens - self.seq_len, self.seq_len):
            chunk = self.tokens[i:i + self.seq_len]
            chunks.append(chunk)
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def make_collate_fn(tokenizer: HuggingFaceTokenizer, seq_len: int):
    def collate(batch: List[torch.Tensor]):
        tokens = torch.stack(batch, dim=0)
        idx = tokens[:, :-1]      
        targets = tokens[:, 1:]   
        
        return {
            "idx": idx,
            "targets": targets,
            "ponder_mask": None,  
        }
    return collate


def train_babylm_tokenizer(
    data_dir: str,
    vocab_size: int = 8192,
    save_path: Optional[str] = None,
) -> HuggingFaceTokenizer:

    data_dir = Path(data_dir)
    train_files = list(data_dir.glob("*.train"))

    for file in TRAIN_FILES:
        if file not in [f.name for f in train_files]:
            raise ValueError(f"Expected training file {file} not found in {data_dir}")

    train_files = [f for f in train_files if f.name in TRAIN_FILES]

    # Use the classmethod to train tokenizer from files
    tokenizer = HuggingFaceTokenizer.train_from_files(
        files=[str(f) for f in train_files],
        vocab_size=vocab_size
    )
    
    # Save tokenizer if save_path is provided
    if save_path:
        save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
        tokenizer.save(save_dir)
    
    return tokenizer


def evaluate_babylm(model, val_dataloader, device, cfg):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    total_expected_steps = 0.0
    
    with torch.no_grad():
        for batch in val_dataloader:
            idx = batch["idx"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            ponder_mask = batch.get("ponder_mask")
            if ponder_mask is not None:
                ponder_mask = ponder_mask.to(device, non_blocking=True)
            
            # Forward pass
            if cfg.use_adaptive_computation:
                loss, _ = model(idx, targets=targets, kv_cache=None,
                               loss_reduction="sum", ponder_mask=ponder_mask)
            else:
                loss, _ = model(idx, targets=targets, kv_cache=None,
                               loss_reduction="sum", ponder_mask=ponder_mask)
            
            batch_tokens = targets.numel()
            total_loss += loss.item()
            total_tokens += batch_tokens
            num_batches += 1
            
            # Track expected steps for ACT models
            if cfg.use_adaptive_computation:
                expected_steps = getattr(model, "last_expected_steps", 0.0)
                total_expected_steps += expected_steps
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    avg_expected_steps = total_expected_steps / num_batches if num_batches > 0 else 0.0
    
    metrics = {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
    }
    
    # Add expected_steps only for ACT models
    if cfg.use_adaptive_computation:
        metrics["val_expected_steps"] = avg_expected_steps
    
    return metrics


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="BabyLM Dataset Preparation")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="directory containing babylm training files")
    parser.add_argument("--vocab_size", type=int, default=8192,
                       help="vacab_size")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json",
                       help="path to save/load the tokenizer")
    parser.add_argument("--seq_len", type=int, default=512,
                       help="")
    parser.add_argument("--train_tokenizer", action="store_true",
                       help="train a new tokenizer")
    
    args = parser.parse_args()
    
    if args.train_tokenizer:
        print("Training tokenizer...")
        tokenizer = train_babylm_tokenizer(
            data_dir=args.data_dir,
            vocab_size=args.vocab_size,
            save_path=args.tokenizer_path,
        )
    else:
        tokenizer = HuggingFaceTokenizer()
        tokenizer = tokenizer.from_directory(args.tokenizer_path)

    dataset = BabyLMDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        split="val",
    )
