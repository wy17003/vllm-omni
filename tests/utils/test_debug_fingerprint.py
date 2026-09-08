# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json

import torch

from vllm_omni.utils.debug_fingerprint import (
    fixed_token_positions,
    gather_paged_tensor_tokens,
    int_sequence_fingerprint,
    tensor_fingerprint,
    tensor_topk_summary,
)


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


def test_fixed_positions_and_paged_gather_follow_logical_block_order() -> None:
    blocks = torch.arange(5 * 4 * 2, dtype=torch.float32).reshape(5, 4, 2)
    positions = fixed_token_positions(seq_len=10, block_size=4)

    assert positions == [0, 1, 2, 3, 4, 5, 7, 8, 9]
    actual = gather_paged_tensor_tokens(blocks, block_ids=[3, 1, 4], token_positions=positions, block_size=4)
    expected_blocks = [3, 3, 3, 3, 1, 1, 1, 4, 4]
    expected = torch.stack([blocks[expected_blocks[i], positions[i] % 4] for i in range(len(positions))])
    torch.testing.assert_close(actual, expected)


def test_tensor_topk_summary_reports_margin() -> None:
    summary = tensor_topk_summary(torch.tensor([-1.0, 3.5, 2.25, 0.0]))

    assert summary == {"token_ids": [1, 2], "scores": [3.5, 2.25], "margin": 1.25}
