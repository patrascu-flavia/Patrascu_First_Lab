# Trained Pavement Crack Segmentation Models

This directory contains the validation-selected checkpoints used in the associated pavement crack segmentation study.

## Models

| Model | Checkpoint | Best Epoch | Best Validation Dice | Final Inference Threshold | Parameters |
|---|---|---:|---:|---:|---:|
| U-Net | `unet_best_by_val_dice.pt` | 195 | 0.6250 | 0.85 | 7,763,041 |
| U-Net++ | `unetpp_best_by_val_dice.pt` | 159 | 0.6244 | 0.85 | 9,159,681 |
| DeepLabV3+ | `deeplabv3plus_best_by_val_dice.pt` | 390 | 0.6082 | 0.80 | 16,587,457 |
| SegFormer | `segformer_best_by_val_dice.pt` | 353 | 0.5595 | 0.85 | 3,714,401 |

## Checkpoint Format

Each `.pt` file is a PyTorch checkpoint dictionary containing:

- `epoch`
- `best_val_dice`
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `config`

The released architecture definitions are provided in the `architectures` directory.

## Loading Models

Use:

```python
from load_model import load_checkpoint_model

model, checkpoint, threshold = load_checkpoint_model(
    "unet_best_by_val_dice.pt",
    device="cpu"
)

print(checkpoint["config"]["model"])
print(threshold)
## Download Trained Checkpoints

The four trained PyTorch checkpoints are archived on Zenodo rather than GitHub because of their file sizes.

https://doi.org/10.5281/zenodo.22697998

Released checkpoints:

- `unet_best_by_val_dice.pt`
- `unetpp_best_by_val_dice.pt`
- `deeplabv3plus_best_by_val_dice.pt`
- `segformer_best_by_val_dice.pt`
