# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json

import torch

from vllm_omni.utils.debug_fingerprint import int_sequence_fingerprint, tensor_fingerprint


def test_int_sequence_fingerprint_uses_canonical_json() -> None:
    values = [17, -2, 300]
    expected = hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()

    assert int_sequence_fingerprint(values) == expected


def test_tensor_fingerprint_supports_full_and_fixed_sampling() -> None:
    tensor = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    full = tensor_fingerprint(tensor)
    sampled = tensor_fingerprint(tensor, sample_count=3)

    assert full["shape"] == [2, 5]
    assert full["sample_numel"] == 10
    assert full["sha256"] == hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()

    expected_sample = tensor.flatten()[torch.tensor([0, 4, 9])]
    assert sampled["sample_numel"] == 3
    assert sampled["sample_first_index"] == 0
    assert sampled["sample_last_index"] == 9
    assert sampled["sha256"] == hashlib.sha256(expected_sample.view(torch.uint8).numpy().tobytes()).hexdigest()
