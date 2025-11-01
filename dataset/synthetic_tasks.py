import random
from dataclasses import dataclass
from typing import Literal, Tuple, List

from torch.utils.data import Dataset


@dataclass
class Difficulty:
    max_operands: int
    max_digits: int
    operations: List[Literal['add', 'mul']]


class AddMul(Dataset):
    
    def __init__(
        self, 
        num_samples: int = 50_000,
        max_operands: int = 6, 
        max_digits: int = 6,   
        seed: int = 0
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_operands = max_operands
        self.max_digits = max_digits
        self.rng = random.Random(seed)
        self.current_difficulty = Difficulty(
            max_operands=max_operands,
            max_digits=max_digits,
            operations=['add', 'mul']
        )
        
    def set_difficulty(self, difficulty: Difficulty):
        """Set difficulty level dynamically. Pass None for random difficulty."""
        self.current_difficulty = difficulty
        
    def __len__(self):
        return self.num_samples
    
    
    def _generate_number(self, num_digits: int) -> int:
        sign = 1 if self.rng.randint(0, 1) == 0 else -1
        num = self.rng.randint(10**(num_digits-1), 10**num_digits - 1)
        return sign * num if num else 0

    def _compute_result(self, operands: List[int], operation: str) -> int:
        result = operands[0]
        for operand in operands[1:]:
            if operation == 'add':
                result += operand
            elif operation == 'mul':
                result *= operand
        return result
    
    def _number_to_digits(self, num: int) -> List[int]:
        if num < 0:
            return ['-'] + [d for d in str(abs(num))]
        return [d for d in str(num)]

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:

        num_operands = self.rng.randint(2, self.current_difficulty.max_operands)
        operation = self.rng.choices(self.current_difficulty.operations, k=1)[0]
        operands = [self._generate_number(self.rng.randint(1, self.current_difficulty.max_digits))
                    for _ in range(num_operands)]
        result = self._compute_result(operands, operation)
        
        input_seq = []
        for i, operand in enumerate(operands):
            if i > 0:
                input_seq.append(operation)
            input_seq.extend(self._number_to_digits(operand))

        output_seq = self._number_to_digits(result)
        return input_seq, output_seq

