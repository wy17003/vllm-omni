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
