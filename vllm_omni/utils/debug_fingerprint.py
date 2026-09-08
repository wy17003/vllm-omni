# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic fingerprints for functional-equivalence diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import torch


def bytes_fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_fingerprint(value: str) -> str:
    return bytes_fingerprint(value.encode("utf-8"))


def int_sequence_fingerprint(values: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in values], separators=(",", ":"))
    return text_fingerprint(payload)


def _evenly_spaced_indices(numel: int, sample_count: int) -> list[int]:
    if sample_count >= numel:
        return list(range(numel))
    if sample_count == 1:
        return [0]
    return [(index * (numel - 1)) // (sample_count - 1) for index in range(sample_count)]


def fixed_token_positions(seq_len: int, block_size: int) -> list[int]:
    """Choose reproducible token positions for paged-cache comparisons."""
    if seq_len <= 0:
        return []
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    candidates = (
        0,
        1,
        block_size - 1,
        block_size,
        2 * block_size - 1,
        2 * block_size,
        seq_len // 4,
        seq_len // 2,
        (3 * seq_len) // 4,
        seq_len - 2,
        seq_len - 1,
    )
    return sorted({position for position in candidates if 0 <= position < seq_len})


def gather_paged_tensor_tokens(
    blocks: torch.Tensor,
    block_ids: Sequence[int],
    token_positions: Sequence[int],
    block_size: int,
) -> torch.Tensor:
    """Gather logical token rows from a paged tensor's physical blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if blocks.ndim < 2:
        raise ValueError("blocks must have block and token dimensions")
    if not token_positions:
        return blocks.new_empty((0, *blocks.shape[2:]))

    physical_blocks: list[int] = []
    offsets: list[int] = []
    for position in token_positions:
        if position < 0:
            raise ValueError("token positions must be non-negative")
        logical_block = position // block_size
        if logical_block >= len(block_ids):
            raise ValueError(f"token position {position} has no block-table entry")
        physical_block = int(block_ids[logical_block])
        if physical_block < 0 or physical_block >= blocks.shape[0]:
            raise ValueError(f"physical block id {physical_block} is out of range")
        physical_blocks.append(physical_block)
        offsets.append(position % block_size)

    block_index = torch.tensor(physical_blocks, dtype=torch.long, device=blocks.device)
    offset_index = torch.tensor(offsets, dtype=torch.long, device=blocks.device)
    return blocks[block_index, offset_index].detach().contiguous()


def tensor_topk_summary(logits: torch.Tensor, k: int = 2) -> dict[str, Any]:
    """Return top-k token IDs, scores, and the top-2 margin for one row."""
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if k <= 0:
        raise ValueError("k must be positive")
    actual_k = min(k, logits.numel())
    values, token_ids = torch.topk(logits, actual_k)
    values_cpu = values.detach().to(device="cpu", dtype=torch.float32)
    token_ids_cpu = token_ids.detach().to(device="cpu", dtype=torch.long)
    scores = [float(value) for value in values_cpu.tolist()]
    result: dict[str, Any] = {
        "token_ids": [int(token_id) for token_id in token_ids_cpu.tolist()],
        "scores": scores,
    }
    result["margin"] = scores[0] - scores[1] if len(scores) >= 2 else None
    return result


def tensor_fingerprint(tensor: torch.Tensor, sample_count: int | None = None) -> dict[str, Any]:
    """Return a stable raw-byte digest and numeric summary for a tensor.

    ``sample_count=None`` fingerprints every element. A positive sample count
    fingerprints fixed, evenly spaced flat positions and is intended for large
    tensors such as KV caches.
    """
    detached = tensor.detach().contiguous()
    numel = detached.numel()
    sampled = detached.reshape(-1)
    sampled_indices: list[int] | None = None
    if sample_count is not None and numel > 0:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        sampled_indices = _evenly_spaced_indices(numel, min(sample_count, numel))
        index_tensor = torch.tensor(sampled_indices, device=sampled.device, dtype=torch.long)
        sampled = sampled.index_select(0, index_tensor)

    cpu_tensor = sampled.to(device="cpu").contiguous()
    raw_bytes = cpu_tensor.view(torch.uint8).numpy().tobytes()
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": numel,
        "sample_numel": cpu_tensor.numel(),
        "sha256": bytes_fingerprint(raw_bytes),
    }
    if sampled_indices is not None:
        result["sample_first_index"] = sampled_indices[0] if sampled_indices else None
        result["sample_last_index"] = sampled_indices[-1] if sampled_indices else None

    if cpu_tensor.numel() > 0:
        stats = cpu_tensor.to(dtype=torch.float32)
        result.update(
            min=float(stats.min().item()),
            max=float(stats.max().item()),
            mean=float(stats.mean().item()),
            std=float(stats.std(unbiased=False).item()),
        )
    return result
