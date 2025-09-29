from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def _prepare_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """Ensure the feature map is three dimensional (C, H, W)."""
    if feature_map.dim() == 4:
        if feature_map.size(0) != 1:
            raise ValueError("Expected a single feature map when dim=4.")
        feature_map = feature_map.squeeze(0)
    if feature_map.dim() != 3:
        raise ValueError("Feature map must have 3 dimensions (C, H, W).")
    return feature_map.contiguous()


def _normalize_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """L2-normalize feature vectors at every spatial location."""

    feature_map = _prepare_feature_map(feature_map)
    c, h, w = feature_map.shape
    flattened = feature_map.view(c, -1).t()  # [HW, C]
    normalized = F.normalize(flattened, dim=1, eps=1e-6)
    return normalized.t().view(c, h, w)


def gram_spatial(feature_map: torch.Tensor, max_positions: int = 64) -> torch.Tensor:
    """Compute the spatial self-similarity Gram matrix."""

    feature_map = _normalize_feature_map(feature_map)
    c, h, w = feature_map.shape
    x = feature_map.view(c, -1).t()  # [HW, C]
    if max_positions is not None and max_positions > 0 and x.size(0) > max_positions:
        indices = torch.randperm(x.size(0), device=x.device)[:max_positions]
        x = x.index_select(0, indices)
    gram = x @ x.t()
    gram = gram / float(c)
    return gram


def gram_channel(feature_map: torch.Tensor) -> torch.Tensor:
    """Compute the channel correlation Gram matrix."""

    feature_map = _normalize_feature_map(feature_map)
    c, h, w = feature_map.shape
    x = feature_map.view(c, -1)  # [C, HW]
    gram = x @ x.t()
    gram = gram / float(h * w)
    return gram


def _flatten_support_labels(
    support_targets: Iterable[Union[Sequence[int], torch.Tensor]],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Flatten the episodic support labels while preserving their order."""

    pieces = []
    for targets in support_targets:
        if torch.is_tensor(targets):
            tensor = targets.to(device=device, dtype=dtype).view(-1)
        else:
            tensor = torch.as_tensor(list(targets), dtype=dtype, device=device).view(-1)
        pieces.append(tensor)
    if not pieces:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.cat(pieces, dim=0)


def assemble_episode_labels(
    query_labels: torch.Tensor,
    support_targets: Iterable[Union[Sequence[int], torch.Tensor]],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Concatenate query and support labels preserving ordering."""

    if query_labels.dim() != 1:
        query_labels = query_labels.view(-1)
    if device is None:
        device = query_labels.device
    support_labels = _flatten_support_labels(
        support_targets, dtype=query_labels.dtype, device=device
    )
    return torch.cat((query_labels.to(device=device), support_labels), dim=0)


def semantic_alignment_loss(feature_list, labels, max_positions: int = 64) -> torch.Tensor:
    """Compute the semantic alignment loss for an episode.

    Args:
        feature_list: list of tensors with shape [C, H, W].
        labels: tensor with the corresponding labels (same length as feature_list).
        max_positions: number of spatial locations to sample for the spatial Gram.

    Returns:
        Scalar tensor representing the semantic alignment loss.
    """

    if not feature_list:
        raise ValueError("feature_list must not be empty")

    device = feature_list[0].device
    labels = labels.to(device)

    spatial_grams = [gram_spatial(f, max_positions=max_positions) for f in feature_list]
    channel_grams = [gram_channel(f) for f in feature_list]

    unique_labels = labels.unique(sorted=True)
    loss = feature_list[0].new_zeros(())
    sample_count = 0

    for lbl in unique_labels:
        idx = torch.nonzero(labels == lbl, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue

        class_spatial = torch.stack([spatial_grams[i] for i in idx], dim=0)
        class_channel = torch.stack([channel_grams[i] for i in idx], dim=0)

        proto_spatial = class_spatial.mean(dim=0)
        proto_channel = class_channel.mean(dim=0)

        spatial_diff = (class_spatial - proto_spatial).reshape(len(idx), -1).mean(dim=1)
        channel_diff = (class_channel - proto_channel).reshape(len(idx), -1).mean(dim=1)

        loss = loss + spatial_diff.sum() + channel_diff.sum()
        sample_count += len(idx)

    if sample_count == 0:
        raise ValueError("semantic_alignment_loss received empty label groups")

    return loss / float(sample_count)


def collect_episode_features(
    feature_pack: Tuple[Sequence[torch.Tensor], Sequence[Sequence[torch.Tensor]]]
):
    """Flatten query and support feature tensors into a single list."""

    query_feats, support_feats = feature_pack
    feature_maps = [feat.squeeze(0) if feat.dim() == 4 else feat for feat in query_feats]
    for class_feat in support_feats:
        feature_maps.extend(
            feat.squeeze(0) if feat.dim() == 4 else feat for feat in class_feat
        )
    return feature_maps


def compute_semantic_alignment(
    feature_pack: Tuple[Sequence[torch.Tensor], Sequence[Sequence[torch.Tensor]]],
    query_labels: torch.Tensor,
    support_targets: Iterable[Union[Sequence[int], torch.Tensor]],
    *,
    max_positions: int = 64,
) -> Tuple[torch.Tensor, int]:
    """Return the semantic alignment loss and the number of feature maps used."""

    feature_maps = collect_episode_features(feature_pack)
    label_tensor = assemble_episode_labels(
        query_labels, support_targets, device=query_labels.device
    )
    if label_tensor.numel() != len(feature_maps):
        raise ValueError("Mismatch between feature and label counts for SA loss")
    loss = semantic_alignment_loss(feature_maps, label_tensor, max_positions=max_positions)
    return loss, len(feature_maps)

