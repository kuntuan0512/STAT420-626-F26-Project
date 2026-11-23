"""Three-slice, mean-query AMCA segmentation shared by M0, M1 and M2."""

from __future__ import annotations

import argparse
import json
import math

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "design_network.py requires PyTorch. Select your PyTorch environment "
        "or install a build appropriate for your CPU/GPU from pytorch.org."
    ) from exc


MODALITIES = ("t1n", "t1c", "t2w", "t2f")
NUM_IMAGE_CHANNELS = 12
NUM_INPUT_CHANNELS = 16
ARCHITECTURE_VERSION = 'amca_mean_query_25d_v1'
NUM_CLASSES = 4
DICE_EPSILON = 1e-6
MODEL_OBJECTIVES = {"M0": "M0", "M1": "M1", "M2": "M2"}


def objective_kind(model_name: str) -> str:
    """Map the architecture ablations to complete/dropout/consistency losses."""
    if not isinstance(model_name, str) or model_name.upper() not in MODEL_OBJECTIVES:
        raise ValueError(f"model_name must be one of {tuple(MODEL_OBJECTIVES)}")
    return MODEL_OBJECTIVES[model_name.upper()]


class DoubleConv(nn.Module):
    """Two same-resolution 3x3 convolutions with instance norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class UpBlock(nn.Module):
    """Learned 2x upsampling, encoder skip concatenation, then DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, skip_channels, kernel_size=2, stride=2
        )
        self.convolutions = DoubleConv(2 * skip_channels, skip_channels)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        upsampled = self.upsample(inputs)
        return self.convolutions(torch.cat((skip, upsampled), dim=1))


def _validate_images(inputs: Tensor, channels: int = NUM_INPUT_CHANNELS) -> None:
    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if inputs.ndim != 4 or inputs.shape[1] != channels:
        raise ValueError(f"inputs must have shape [batch, {channels}, height, width]")
    if inputs.shape[0] < 1 or min(inputs.shape[-2:]) < 1:
        raise ValueError("inputs must have a nonempty batch and spatial dimensions")
    if not inputs.is_floating_point():
        raise TypeError("inputs must use a floating-point dtype")


class UNet25D(nn.Module):
    """Four-level 2D U-Net predicting the center of a three-slice input.

    Spatial sizes need not be divisible by 16. Right/bottom replication padding
    makes all skip connections exact, and the logits are cropped back to the
    original size. Height and width must each be at least 32 so instance norm
    has more than one spatial element at the deepest level.
    """

    def __init__(self, base_channels: int = 16, in_channels: int = NUM_INPUT_CHANNELS) -> None:
        super().__init__()
        if isinstance(base_channels, bool) or not isinstance(base_channels, int):
            raise TypeError("base_channels must be an integer")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        self.base_channels = base_channels
        self.in_channels = in_channels
        self.out_classes = NUM_CLASSES
        channels = [base_channels * (2**level) for level in range(5)]
        self.encoder = nn.ModuleList(
            [
                DoubleConv(in_channels, channels[0]),
                DoubleConv(channels[0], channels[1]),
                DoubleConv(channels[1], channels[2]),
                DoubleConv(channels[2], channels[3]),
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(channels[3], channels[4])
        self.decoder = nn.ModuleList(
            [UpBlock(channels[level + 1], channels[level]) for level in (3, 2, 1, 0)]
        )
        self.classifier = nn.Conv2d(channels[0], NUM_CLASSES, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_images(inputs, self.in_channels)
        height, width = inputs.shape[-2:]
        if min(height, width) < 32:
            raise ValueError("U-Net requires height and width >= 32")
        pad_height, pad_width = (-height) % 16, (-width) % 16
        features = F.pad(inputs, (0, pad_width, 0, pad_height), mode="replicate")
        skips = []
        for encoder in self.encoder:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)
        features = self.bottleneck(features)
        for decoder, skip in zip(self.decoder, reversed(skips)):
            features = decoder(features, skip)
        return self.classifier(features)[..., :height, :width]

    def probabilities(self, inputs: Tensor) -> Tensor:
        """Return class probabilities in float32 (does not change train/eval mode)."""
        return F.softmax(self(inputs).float(), dim=1)


class AvailabilityMaskedCrossAttention(nn.Module):
    """At each pixel, the available-stream mean queries four modality keys.

    Modality embeddings enter projected keys only. Missing keys are excluded
    before softmax. Values contain image features, without identity embeddings.
    Four heads and the residual implement the supplied methodology directly.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = 4
        if channels < self.heads or channels % self.heads:
            raise ValueError("Fusion channels must be a positive multiple of four")
        self.channels = channels
        self.head_dimension = channels // self.heads
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.modality_embedding = nn.Parameter(torch.empty(4, channels))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.output_projection = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, streams: Tensor, available: Tensor) -> Tensor:
        batch, modalities, channels, height, width = streams.shape
        gate = available[:, :, None, None, None]
        streams = torch.where(gate, streams, 0.0)
        mean = streams.sum(dim=1) / available.sum(dim=1)[:, None, None, None]
        query = self.query(mean).reshape(batch, self.heads, self.head_dimension, height, width)
        flattened = streams.reshape(batch * modalities, channels, height, width)
        keys = self.key(flattened).reshape(batch, modalities, self.heads,
                                           self.head_dimension, height, width)
        embedding = self.modality_embedding.reshape(1, modalities, self.heads,
                                                    self.head_dimension, 1, 1)
        keys = keys + embedding
        values = self.value(flattened).reshape(batch, modalities, self.heads,
                                               self.head_dimension, height, width)
        # Keep the dot product in float32 even inside a CUDA autocast region.
        with torch.autocast(device_type=streams.device.type, enabled=False):
            logits = torch.einsum("bhdxy,bmhdxy->bhmxy", query.float(), keys.float())
        logits = logits / math.sqrt(self.head_dimension)
        logits = logits.masked_fill(~available[:, None, :, None, None], -torch.inf)
        weights = logits.softmax(dim=2).to(values.dtype)
        attended = torch.einsum("bhmxy,bmhdxy->bhdxy", weights, values)
        attended = attended.reshape(batch, channels, height, width)
        return mean + self.output_projection(attended)


class ModalityFusionUNet25D(nn.Module):
    """Identical three-slice stems, mean-query fusion and U-Net for M0--M2.

    The optional concatenation ablation retains the stems and concatenates
    their gated feature maps directly before the U-Net's first encoder block.
    """
    def __init__(self, base_channels: int = 16, fusion_mode: str = "amca") -> None:
        super().__init__()
        if fusion_mode not in {"amca", "concatenation"}:
            raise ValueError("fusion_mode must be amca or concatenation")
        self.base_channels = base_channels
        self.fusion_mode = fusion_mode
        self.fusion_channels = 4 * max(1, base_channels // 8)
        self.attention_heads = 4
        self.stems = nn.ModuleList([DoubleConv(3, self.fusion_channels) for _ in MODALITIES])
        if fusion_mode == "amca":
            self.fusion = AvailabilityMaskedCrossAttention(self.fusion_channels)
        fused_channels = self.fusion_channels if fusion_mode == "amca" else 4 * self.fusion_channels
        self.backbone = UNet25D(base_channels, in_channels=fused_channels + 4)

    @staticmethod
    def availability(inputs: Tensor) -> Tensor:
        maps = inputs[:, NUM_IMAGE_CHANNELS:]
        if bool(((maps != 0) & (maps != 1)).any()):
            raise ValueError("Availability maps must be binary")
        available = maps[:, :, 0, 0].bool()
        if not torch.equal(maps.bool(), available[:, :, None, None].expand_as(maps)):
            raise ValueError("Availability maps must be constant within each slice")
        if not bool(available.any(dim=1).all()):
            raise ValueError("At least one MRI modality must be available per sample")
        return available

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_images(inputs)
        available = self.availability(inputs)
        features = []
        for index, stem in enumerate(self.stems):
            gate = available[:, index, None, None, None]
            images = torch.where(gate, inputs[:, 3*index:3*index+3], 0.0)
            features.append(stem(images) * gate.to(inputs.dtype))
        if self.fusion_mode == "amca":
            fused = self.fusion(torch.stack(features, dim=1), available)
        else:
            fused = torch.cat(features, dim=1)
        return self.backbone(torch.cat((fused, inputs[:, NUM_IMAGE_CHANNELS:]), dim=1))

    def probabilities(self, inputs: Tensor) -> Tensor:
        return F.softmax(self(inputs).float(), dim=1)


def build_model(model_name: str, base_channels: int = 16, fusion_mode: str = "amca") -> nn.Module:
    """Select the objective name while keeping a common fusion architecture."""
    objective_kind(model_name)
    return ModalityFusionUNet25D(base_channels, fusion_mode)


def drop_one_modality(inputs: Tensor, missing_indices: Tensor | int) -> Tensor:
    """Return a copy with one MRI modality and its availability map zeroed.

    ``missing_indices`` may be a single integer in [0, 3] for the whole batch
    or a one-dimensional integer tensor of length ``batch``. All three image
    channels belonging to each sample's selected modality is removed.
    The source tensor is never modified.
    """
    _validate_images(inputs)
    batch = inputs.shape[0]
    if isinstance(missing_indices, int) and not isinstance(missing_indices, bool):
        if not 0 <= missing_indices < len(MODALITIES):
            raise ValueError("missing modality index must be in [0, 3]")
        indices = torch.full(
            (batch,), missing_indices, dtype=torch.long, device=inputs.device
        )
    elif isinstance(missing_indices, Tensor):
        if missing_indices.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise TypeError("missing_indices tensor must have an integer dtype")
        if missing_indices.ndim != 1 or missing_indices.shape[0] != batch:
            raise ValueError("missing_indices must have shape [batch]")
        indices = missing_indices.to(device=inputs.device, dtype=torch.long)
        if bool(((indices < 0) | (indices >= len(MODALITIES))).any()):
            raise ValueError("every missing modality index must be in [0, 3]")
    else:
        raise TypeError("missing_indices must be an integer or integer tensor")
    channel_mask = torch.ones(
        (batch, NUM_INPUT_CHANNELS), device=inputs.device, dtype=inputs.dtype
    )
    rows = torch.arange(batch, device=inputs.device)
    image_indices = 3 * indices[:, None] + torch.arange(3, device=inputs.device)
    channel_mask[rows[:, None], image_indices] = 0
    channel_mask[rows, NUM_IMAGE_CHANNELS + indices] = 0
    return inputs * channel_mask[:, :, None, None]


def _validate_logits(logits: Tensor, name: str = "logits") -> None:
    if not isinstance(logits, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"{name} must have shape [batch, 4, height, width]")
    if logits.shape[0] < 1 or min(logits.shape[-2:]) < 1:
        raise ValueError(f"{name} must have nonempty dimensions")
    if not logits.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def segmentation_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy plus mean soft-Dice loss for exclusive tumor classes 1..3.

    Dice sums over all voxels in the batch before averaging the three classes,
    as specified in the proposal. Cross-entropy includes background. Both terms
    use float32 for stability during mixed-precision training.
    """
    _validate_logits(logits)
    if not isinstance(target, Tensor):
        raise TypeError("target must be a torch.Tensor")
    if target.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ValueError("target must have shape [batch, height, width]")
    if target.device != logits.device:
        raise ValueError("target and logits must be on the same device")
    if target.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    ):
        raise TypeError("target must contain integer class labels")
    if bool(((target < 0) | (target >= NUM_CLASSES)).any()):
        raise ValueError("target labels must be 0, 1, 2, or 3; remap raw labels first")
    labels = target.long()
    stable_logits = logits.float()
    # Preserve the saved Colab implementation's deterministic CUDA loss path.
    # Flatten pixels into classification rows: the objective is unchanged.
    pixel_logits = stable_logits.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
    cross_entropy = F.cross_entropy(pixel_logits, labels.reshape(-1))
    probabilities = F.softmax(stable_logits, dim=1)[:, 1:]
    one_hot = F.one_hot(labels, num_classes=NUM_CLASSES).permute(0, 3, 1, 2)
    tumor_targets = one_hot[:, 1:].to(dtype=torch.float32)
    sum_axes = (0, 2, 3)
    intersection = (probabilities * tumor_targets).sum(dim=sum_axes)
    denominator = probabilities.sum(dim=sum_axes) + tumor_targets.sum(dim=sum_axes)
    dice = (2.0 * intersection + DICE_EPSILON) / (denominator + DICE_EPSILON)
    return cross_entropy + 1.0 - dice.mean()


def consistency_loss(full_logits: Tensor, missing_logits: Tensor, brain_mask: Tensor) -> Tensor:
    """Mean KL(stop-gradient full probabilities || missing probabilities).

    Only brain voxels participate. The mask is [batch, height, width] or
    [batch, 1, height, width], with boolean or binary numeric values. Compute it
    from the original complete scans, before normalization or modality removal.
    The full-input teacher is detached for this term only; its segmentation
    loss still trains the shared network. An entirely empty mask is an error.
    """
    _validate_logits(full_logits, "full_logits")
    _validate_logits(missing_logits, "missing_logits")
    if full_logits.shape != missing_logits.shape:
        raise ValueError("full_logits and missing_logits must have identical shapes")
    if full_logits.device != missing_logits.device:
        raise ValueError("full_logits and missing_logits must be on the same device")
    if not isinstance(brain_mask, Tensor):
        raise TypeError("brain_mask must be a torch.Tensor")
    if brain_mask.ndim == 4 and brain_mask.shape[1] == 1:
        brain_mask = brain_mask[:, 0]
    expected_shape = (full_logits.shape[0], *full_logits.shape[-2:])
    if brain_mask.shape != expected_shape:
        raise ValueError("brain_mask must have shape [batch, height, width]")
    if brain_mask.device != full_logits.device:
        raise ValueError("brain_mask and logits must be on the same device")
    if bool(((brain_mask != 0) & (brain_mask != 1)).any()):
        raise ValueError("brain_mask must contain only binary 0/1 or boolean values")
    valid_voxels = brain_mask.bool()
    if not bool(valid_voxels.any()):
        raise ValueError("brain_mask is empty; consistency learning needs brain voxels")
    teacher_log = F.log_softmax(full_logits.detach().float(), dim=1)
    student_log = F.log_softmax(missing_logits.float(), dim=1)
    # F.kl_div(input=student_log, target=teacher_log, log_target=True) computes
    # KL(teacher || student); reduction is explicit to average voxels, not classes.
    per_voxel_kl = F.kl_div(
        student_log, teacher_log, reduction="none", log_target=True
    ).sum(dim=1)
    return per_voxel_kl.masked_select(valid_voxels).mean()


def model_objective(
    model_name: str,
    full_logits: Tensor,
    target: Tensor,
    missing_logits: Tensor | None = None,
    brain_mask: Tensor | None = None,
    consistency_weight: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Return the exact M0/M1/M2 objective plus detached logging scalars.

    M0: segmentation(full).
    M1: 0.5 * [segmentation(full) + segmentation(single-missing)].
    M2: M1 + consistency_weight * KL(full teacher || missing student).

    Training controls forward passes and selects one missing modality per sample.
    Logging keys are loss, segmentation, segmentation_full,
    segmentation_missing, consistency, and consistency_weight.
    """
    model_name = objective_kind(model_name)
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("consistency_weight must be finite and nonnegative")
    full_segmentation = segmentation_loss(full_logits, target)
    zero = full_segmentation.new_zeros(())
    missing_segmentation, consistency = zero, zero
    segmentation = full_segmentation
    if model_name in {"M1", "M2"}:
        if missing_logits is None:
            raise ValueError(f"{model_name} requires missing_logits")
        _validate_logits(missing_logits, "missing_logits")
        if full_logits.shape != missing_logits.shape:
            raise ValueError("full_logits and missing_logits must have identical shapes")
        missing_segmentation = segmentation_loss(missing_logits, target)
        segmentation = 0.5 * (full_segmentation + missing_segmentation)
    if model_name == "M2":
        if brain_mask is None:
            raise ValueError("M2 requires brain_mask")
        consistency = consistency_loss(full_logits, missing_logits, brain_mask)
    total = segmentation + (consistency_weight * consistency if model_name == "M2" else zero)
    scalars = {
        "loss": float(total.detach()),
        "segmentation": float(segmentation.detach()),
        "segmentation_full": float(full_segmentation.detach()),
        "segmentation_missing": float(missing_segmentation.detach()),
        "consistency": float(consistency.detach()),
        "consistency_weight": float(consistency_weight if model_name == "M2" else 0.0),
    }
    return total, scalars


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def self_test():
    from verify_amca import run_checks
    return run_checks()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="Run small CPU correctness tests")
    parser.add_argument("--base-channels", type=int, default=16, help="U-Net base width")
    parser.add_argument("--model", choices=list(MODEL_OBJECTIVES), default="M2")
    parser.add_argument("--height", type=int, default=64, help="Synthetic example height (>=32)")
    parser.add_argument("--width", type=int, default=64, help="Synthetic example width (>=32)")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    model = build_model(args.model, base_channels=args.base_channels).eval()
    inputs = torch.randn(1, NUM_INPUT_CHANNELS, args.height, args.width)
    inputs[:, NUM_IMAGE_CHANNELS:] = 1.0
    with torch.no_grad():
        logits = model(inputs)
    print(model)
    print(json.dumps({
        "architecture": type(model).__name__,
        "model": args.model,
        "fusion_mode": getattr(model, "fusion_mode", "early_fusion"),
        "encoder_channels": [args.base_channels * (2**level) for level in range(4)],
        "bottleneck_channels": args.base_channels * 16,
        "input_shape": list(inputs.shape),
        "output_logits_shape": list(logits.shape),
        "trainable_parameters": count_parameters(model),
        "simplified_study_width": args.base_channels == 16,
    }, indent=2))


if __name__ == "__main__":
    main()
