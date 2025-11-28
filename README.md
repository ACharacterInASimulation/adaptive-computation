# Adaptive Computation for Transformers

This repository implements adaptive transformer layers (ACT / Ponder-style) with shared KV-cache, trained on:
- Synthetic arithmetic dataset (addition & multiplication)
- BabyLM language modeling dataset

## Repository Structure

- `train.py`: Main training script (loads config, builds model, trains & evaluates)
- `train_earlyexit.py`: Training script for models with early-exit heads
- `configs/`: YAML configs for different models/datasets (e.g. `babylm_act.yaml`, `ponder.yaml`)
- `models/`: Transformer blocks, ACT/Ponder modules, early-exit heads, shared KV-cache
- `dataset/`: Dataset loaders for synthetic arithmetic and BabyLM
- `scripts/`: Helper scripts (e.g. `download_babylm.sh` for dataset download)
- `Notebooks/`: Demo Jupyter notebooks with sample runs, inputs and outputs

## Dependencies & Installation

Tested with **Python ≥ 3.9**.

Clone and install:
```bash
git clone https://github.com/ACharacterInASimulation/adaptive-computation.git
cd adaptive-computation
pip install -r requirements.txt
```

## Data

### BabyLM (public dataset)
Download using the provided script:
```bash
./scripts/download_babylm.sh
```
This downloads BabyLM from publicly available links into the expected `data/` folder (see the script for exact paths).

### Synthetic arithmetic dataset
The synthetic addition/multiplication data is generated on-the-fly inside the dataset loader.

## How to Run

**BabyLM + ACT model:**
```bash
python train.py --configs ./configs/babylm_act.yaml
```

**Synthetic arithmetic + Ponder model:**
```bash
python train.py --configs ./configs/ponder.yaml
```

All available experiment configs are in `./configs`.