# adaptive-computation


Currently supported features include:

- adaptive transformer (ACT/Ponder) layers with shared KV cache

- Synthetic dataset - Addition multiplication
- Babylm dataset


## Download Babylm Dataset:

`./scripts/download_babylm.sh`

## Train
### babylm dataset act
`python train.py --configs ./configs/babylm_act.yaml`
### synthetic dataset ponder
`python train.py --configs ./configs/ponder.yaml`

### All the available configs - ./configs




