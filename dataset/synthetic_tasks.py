# dataset_addmul.py
import random
from dataclasses import dataclass
from typing import List, Tuple, Literal
from torch.utils.data import Dataset
import torch

@dataclass
class Difficulty:
    max_operands: int
    max_digits: int
    operations: List[Literal["+", "*"]]

class AddMul(Dataset):
    def __init__(self, num_samples=50_000, max_operands=6, max_digits=6, seed=0, operations=["+", "*"]):
        super().__init__()
        self.num_samples = num_samples
        self.max_operands = max_operands
        self.max_digits = max_digits
        self.rng = random.Random(seed)
        self.current_difficulty = Difficulty(
            max_operands=max_operands, max_digits=max_digits, operations=operations
        )

    def set_difficulty(self, difficulty: Difficulty):
        self.current_difficulty = difficulty

    def __len__(self):
        return self.num_samples

    def _generate_number(self, num_digits: int) -> int:
        sign = 1 if self.rng.randint(0, 1) == 0 else -1
        lo = 10 ** (num_digits - 1)
        hi = 10 ** num_digits - 1
        n = self.rng.randint(lo, hi)
        return sign * n

    def _compute_result(self, operands: List[int], op: str) -> int:
        assert op in ["+", "*"]
        result = operands[0]
        for t in operands[1:]:
            result = result + t if op == "+" else result * t
        return result

    def _number_to_tokens(self, n: int) -> List[str]:
        if n < 0:
            return ["-"] + list(str(abs(n)))
        return list(str(n))

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str]]:
        num_operands = self.rng.randint(2, self.current_difficulty.max_operands)
        op = self.rng.choice(self.current_difficulty.operations)
        operands = [
            self._generate_number(self.rng.randint(1, self.current_difficulty.max_digits))
            for _ in range(num_operands)
        ]
        result = self._compute_result(operands, op)

        input_seq: List[str] = []
        for i, val in enumerate(operands):
            if i > 0:
                input_seq.append(op)    
            input_seq.extend(self._number_to_tokens(val))
        input_seq.append("=")           

        output_seq = self._number_to_tokens(result) 
        return input_seq, output_seq



def make_collate_fn(tokenizer, max_seq_len: int):
    pad_id = tokenizer.pad_id
    eos_tok = tokenizer.EOS  

    def collate(batch):
        Xs, Ys, Ps = [], [], []
        max_T = 0

        for input_seq, output_seq in batch:
            S = input_seq + output_seq + [eos_tok] 
            X = S[:-1]
            Y = S[1:]

            '''1+2=3<eos>
            -1 -1 -1 3 <eos>
            '''

            L_in = len(input_seq)       
            Y_ids = tokenizer.encode(Y)
            labels = [-1] * (L_in - 1) + Y_ids[(L_in - 1):]   ## output from '=' is supervised 
            ponder = [0] * (L_in - 1) + [1] * (len(X) - (L_in - 1))

            X_ids = tokenizer.encode(X)

            if len(X_ids) > max_seq_len:
                raise ValueError(f"Sample len {len(X_ids)} > max_seq_len={max_seq_len}")

            Xs.append(X_ids)
            Ys.append(labels)
            Ps.append(ponder)
            max_T = max(max_T, len(X_ids))

        B, T = len(batch), max_T
        idx = torch.full((B, T), pad_id, dtype=torch.long)
        targets = torch.full((B, T), -1, dtype=torch.long)
        ponder_mask = torch.zeros((B, T), dtype=torch.float32)

        for b in range(B):
            L = len(Xs[b])
            idx[b, :L] = torch.tensor(Xs[b], dtype=torch.long)
            targets[b, :L] = torch.tensor(Ys[b][:L], dtype=torch.long)  
            ponder_mask[b, :L] = torch.tensor(Ps[b][:L], dtype=torch.float32)

        return {"idx": idx, "targets": targets, "ponder_mask": ponder_mask}

    return collate


class Tokenizer:
    """
    Vocab: <pad>, <eos>, 0-9, +, -, *, =
    """
    def __init__(self):
        self.PAD = "<pad>"
        self.EOS = "<eos>"
        symbols = [self.PAD, self.EOS] + list("0123456789") + ["+", "-", "*", "="]
        self.stoi = {s: i for i, s in enumerate(symbols)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.pad_id = self.stoi[self.PAD]
        self.eos_id = self.stoi[self.EOS]

    def __len__(self):
        return len(self.stoi)

    def encode(self, seq):  
        return [self.stoi[s] for s in seq]

    def decode(self, ids):  
        return [self.itos[i] for i in ids]


def _safe_int(s):
    try:
        if s == "" or s == "-":
            return None
        return int(s)
    except Exception:
        return None

def generate_eval(model, ds, tokenizer, n_samples=1000, max_tokens=40, temperature=0.0):
    """
    Evaluate model on arithmetic tasks.
    
    Returns:
        dict with metrics: exact_match_acc, digit_acc, avg_steps, normalized_error
    """
    model.eval()
    total_steps = 0
    n_tokens = 0
    exact_match = 0
    total_digits_correct = 0
    total_digits = 0
    normalized_error = 0.0
    n_valid = 0  

    for idx in range(min(n_samples, len(ds))):
        input_seq, output_seq = ds[idx]
        prompt_ids = tokenizer.encode(input_seq)  
        ponder_mask = [0] * len(prompt_ids)
        ponder_mask[-1] = 1  # Only ponder on the '=' token
        
        outs = []
        for tok, n_steps in model.generate(tokens=prompt_ids, ponder_mask=ponder_mask, 
                                          max_tokens=max_tokens, temperature=temperature):
            if tok == tokenizer.eos_id:
                break
            total_steps += n_steps
            n_tokens += 1
            outs.append(tok)
        
        output_str = "".join(tokenizer.decode(outs)).strip()
        gold_str = "".join(output_seq).strip()
        
        # Convert to integers for numerical comparison
        output_int = _safe_int(output_str)
        gold_int = _safe_int(gold_str)

        # Exact match check
        if output_int is not None and gold_int is not None:
            n_valid += 1
            if output_int == gold_int:
                exact_match += 1
            
            
            output_digits = str(output_int)
            gold_digits = str(gold_int)
            total_digits += len(gold_digits)
            
            
            for i in range(min(len(output_digits), len(gold_digits))):
                if output_digits[i] == gold_digits[i]:
                    total_digits_correct += 1
            
            # |predicted - gold| / |gold|
            if gold_int != 0:
                normalized_error += abs(gold_int - output_int) / abs(gold_int)
            elif output_int != 0:
                normalized_error += 1.0 
        else:
            n_valid += 1
            normalized_error += 1.0
            if gold_int is not None:
                total_digits += len(str(gold_int))

    # Compute metrics
    exact_match_acc = exact_match / n_valid if n_valid > 0 else 0.0
    digit_acc = total_digits_correct / total_digits if total_digits > 0 else 0.0
    avg_steps = total_steps / n_tokens if n_tokens > 0 else 0.0
    avg_normalized_error = normalized_error / n_valid if n_valid > 0 else 1.0

    return {
        "exact_match_acc": exact_match_acc * 100,
        "digit_acc": digit_acc * 100,
        "avg_steps": avg_steps,
        "normalized_error": avg_normalized_error * 100,
        "n_samples": n_valid,
        "n_tokens": n_tokens,
    }


