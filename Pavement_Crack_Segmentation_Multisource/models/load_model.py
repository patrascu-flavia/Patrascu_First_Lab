from pathlib import Path
import sys
import torch

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent
ARCH_DIR = MODEL_DIR / "architectures"

sys.path.insert(0, str(ARCH_DIR))

# ---------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------

from unet import UNet
from unetpp import UNetPlusPlus
from deeplabv3plus import DeepLabV3Plus
from segformer import SegFormer


# ---------------------------------------------------------------------
# Final validation-tuned inference thresholds used for evaluation
# ---------------------------------------------------------------------

FINAL_THRESHOLDS = {
    "U-Net": 0.85,
    "U-Net++": 0.85,
    "DeepLabV3+": 0.80,
    "SegFormer": 0.85,
}


def build_model_from_config(config):
    """
    Construct the exact architecture associated with a saved checkpoint.

    Parameters
    ----------
    config : dict
        Configuration dictionary stored inside the training checkpoint.

    Returns
    -------
    torch.nn.Module
        Instantiated segmentation model.
    """

    model_name = config["model"]

    if model_name == "U-Net":

        model = UNet(
            in_channels=3,
            out_channels=1,
            base_channels=config.get("base_channels", 32),
        )

    elif model_name == "U-Net++":

        model = UNetPlusPlus(
            in_channels=3,
            out_channels=1,
            base_channels=config.get("base_channels", 32),
        )

    elif model_name == "DeepLabV3+":

        model = DeepLabV3Plus(
            in_channels=3,
            out_channels=1,
            base_channels=config.get("base_channels", 32),
        )

    elif model_name == "SegFormer":

        model = SegFormer(
            in_channels=3,
            out_channels=1,
            embed_dims=config.get(
                "embed_dims",
                [32, 64, 160, 256],
            ),
            num_heads=config.get(
                "num_heads",
                [1, 2, 5, 8],
            ),
            depths=config.get(
                "depths",
                [2, 2, 2, 2],
            ),
            sr_ratios=config.get(
                "sr_ratios",
                [8, 4, 2, 1],
            ),
            mlp_ratio=config.get(
                "mlp_ratio",
                4,
            ),
            decoder_dim=config.get(
                "decoder_dim",
                256,
            ),
        )

    else:
        raise ValueError(
            f"Unsupported model name in checkpoint: {model_name}"
        )

    return model


def load_checkpoint_model(
    checkpoint_path,
    device="cpu",
):
    """
    Load one of the released pavement crack segmentation checkpoints.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Path to the .pt checkpoint.

    device : str
        PyTorch device, for example 'cpu' or 'cuda'.

    Returns
    -------
    model : torch.nn.Module
        Model with checkpoint weights loaded.

    checkpoint : dict
        Original checkpoint dictionary.

    threshold : float
        Final validation-tuned inference threshold.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain a 'config' dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    config = checkpoint["config"]

    model = build_model_from_config(config)

    # strict=True ensures exact architectural compatibility.
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    model_name = config["model"]

    threshold = FINAL_THRESHOLDS.get(
        model_name,
        config.get("threshold", 0.5),
    )

    return model, checkpoint, threshold


def predict_binary_mask(
    model,
    image_tensor,
    threshold,
):
    """
    Convert model logits into a binary pavement crack mask.

    image_tensor must have shape:
        [B, 3, H, W]

    Input values are expected to be scaled to [0, 1].
    """

    with torch.inference_mode():

        logits = model(image_tensor)

        probabilities = torch.sigmoid(logits)

        mask = (
            probabilities >= threshold
        ).to(torch.uint8)

    return mask


if __name__ == "__main__":

    print(
        "Import load_checkpoint_model() "
        "to load a released checkpoint."
    )