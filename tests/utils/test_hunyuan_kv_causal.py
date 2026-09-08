# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import pytest
import torch

from vllm_omni.utils.hunyuan_kv_causal import HunyuanKVCausalProbe

PROMPT = list(range(19))
P_BLOCKS = [4, 1, 6, 2, 7]
D_BLOCKS = [3, 8, 0, 5, 9]
BLOCK_SIZE = 4


def _caches(dtype=torch.bfloat16):
    producer, consumer = [], []
    for layer in range(3):
        p_pair, d_pair = [], []
        for kind in range(2):
            p = (torch.arange(12 * 4 * 2 * 3).reshape(12, 4, 2, 3) / 100 + layer + kind).to(dtype)
            d = torch.full_like(p, -100)
            for position in range(len(PROMPT)):
                d[D_BLOCKS[position // 4], position % 4].copy_(p[P_BLOCKS[position // 4], position % 4])
            d[D_BLOCKS[-1], 2].add_(1)  # Emulate prompt-tail recomputation.
            p_pair.append(p)
            d_pair.append(d)
        producer.append(tuple(p_pair))
        consumer.append(tuple(d_pair))
    return producer, consumer


def _copy(layers):
    return [tuple(t.clone() for t in pair) for pair in layers]


def _equal(actual, expected):
    for actual_pair, expected_pair in zip(actual, expected, strict=True):
        for a, e in zip(actual_pair, expected_pair, strict=True):
            assert torch.equal(a, e)


def _publish(tmp_path, layers, rank=0, request="req", mode="restore"):
    p = HunyuanKVCausalProbe(mode, str(tmp_path), "kv_producer", rank, 4)
    p.after_forward(request, PROMPT, 0, 19, layers, P_BLOCKS, 4)
    assert not p._path(request).exists()  # D must not see a partial snapshot.
    p.after_sample([request], torch.tensor([[791]]))
    assert p._path(request).is_file()
    p.finish([request])  # Producer completion must not delete D's reference.
    return p


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["check", "restore"])
def test_intervention_only_changes_tail_across_physical_block_layouts(tmp_path, caplog, rank, dtype, mode):
    caplog.set_level(logging.INFO)
    producer, consumer = _caches(dtype)
    original_producer, expected = _copy(producer), _copy(consumer)
    _publish(tmp_path, producer, rank, mode=mode)
    if mode == "restore":
        for p_pair, e_pair in zip(producer, expected, strict=True):
            for p, e in zip(p_pair, e_pair, strict=True):
                e[D_BLOCKS[-1], 2].copy_(p[P_BLOCKS[-1], 2])
    d = HunyuanKVCausalProbe(mode, str(tmp_path), "kv_consumer", rank, 4)
    d.after_forward("req", PROMPT, 18, 1, consumer, D_BLOCKS, 4)
    _equal(consumer, expected)  # Includes prefix, padding and unrelated blocks.
    _equal(producer, original_producer)
    tokens = torch.tensor([[791]])
    d.after_sample(["req"], tokens)
    assert tokens.tolist() == [[791]]
    assert not d._path("req").exists()
    # Later decode steps cannot trigger another intervention or file read.
    d.after_forward("req", PROMPT, 19, 1, consumer, D_BLOCKS, 4)
    d.after_sample(["req"], torch.tensor([[1217]]))
    assert '"event":"prefix_verified"' in caplog.text
    assert ('"event":"tail_restored"' in caplog.text) == (mode == "restore")
    assert '"equal":true' in caplog.text
    d.finish(["req"])
    assert not d.done


def test_unsampled_prefix_corruption_in_last_layer_rejects_before_any_write(tmp_path):
    producer, consumer = _caches()
    _publish(tmp_path, producer)
    consumer[-1][1][D_BLOCKS[6 // 4], 6 % 4, 0, 0].add_(1)
    before = _copy(consumer)
    d = HunyuanKVCausalProbe("restore", str(tmp_path), "kv_consumer", 0, 4)
    with pytest.raises(RuntimeError, match="2/value: prefix_sha256 mismatch"):
        d.after_forward("req", PROMPT, 18, 1, consumer, D_BLOCKS, 4)
    _equal(consumer, before)


@pytest.mark.parametrize("field,value", [("request_id", "other"), ("tp_rank", 3), ("tp_size", 2), ("prompt_len", 20)])
def test_wrong_snapshot_identity_rejected(tmp_path, field, value):
    producer, consumer = _caches()
    p = _publish(tmp_path, producer)
    snapshot = torch.load(p._path("req"), weights_only=True)
    snapshot["metadata"][field] = value
    torch.save(snapshot, p._path("req"))
    before = _copy(consumer)
    d = HunyuanKVCausalProbe("restore", str(tmp_path), "kv_consumer", 0, 4)
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        d.after_forward("req", PROMPT, 18, 1, consumer, D_BLOCKS, 4)
    _equal(consumer, before)


def test_corrupt_tail_in_last_layer_rejects_before_any_write(tmp_path):
    producer, consumer = _caches()
    p = _publish(tmp_path, producer)
    snapshot = torch.load(p._path("req"), weights_only=True)
    snapshot["tails"][-1].add_(1)
    torch.save(snapshot, p._path("req"))
    before = _copy(consumer)
    d = HunyuanKVCausalProbe("restore", str(tmp_path), "kv_consumer", 0, 4)
    with pytest.raises(RuntimeError, match="tail checksum mismatch"):
        d.after_forward("req", PROMPT, 18, 1, consumer, D_BLOCKS, 4)
    _equal(consumer, before)


def test_first_token_mismatch_invalidates_experiment_and_retains_snapshot(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    producer, consumer = _caches()
    _publish(tmp_path, producer)
    d = HunyuanKVCausalProbe("restore", str(tmp_path), "kv_consumer", 0, 4)
    d.after_forward("req", PROMPT, 18, 1, consumer, D_BLOCKS, 4)
    with pytest.raises(RuntimeError, match="consumer first token 42 != producer 791"):
        d.after_sample(["req"], torch.tensor([[42]]))
    assert '"equal":false' in caplog.text
    assert d._path("req").exists()


@pytest.mark.parametrize("computed,scheduled", [(0, 19), (17, 2), (19, 1)])
def test_wrong_consumer_schedule_rejected(tmp_path, computed, scheduled):
    _, consumer = _caches()
    d = HunyuanKVCausalProbe("restore", str(tmp_path), "kv_consumer", 0, 4)
    with pytest.raises(RuntimeError, match="first schedule"):
        d.after_forward("req", PROMPT, computed, scheduled, consumer, D_BLOCKS, 4)


def test_non_pd_logs_full_hashes_without_creating_snapshot_or_mutating_cache(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    producer, _ = _caches()
    before = _copy(producer)
    probe = HunyuanKVCausalProbe("check", str(tmp_path), "non_pd", 0, 4)
    probe.after_forward("req", PROMPT, 0, 19, producer, P_BLOCKS, 4)
    probe.after_sample(["req"], torch.tensor([[791]]))
    _equal(producer, before)
    assert not list(tmp_path.iterdir())
    assert caplog.text.count('"prefix_sha256"') == 6


def test_missing_or_stale_snapshot_fails_explicitly(tmp_path):
    producer, consumer = _caches()
    d = HunyuanKVCausalProbe("restore", str(tmp_path), "kv_consumer", 0, 4)
    with pytest.raises(RuntimeError, match="snapshot missing"):
        d.after_forward("req", PROMPT, 18, 1, consumer, D_BLOCKS, 4)
    _publish(tmp_path, producer)
    p = HunyuanKVCausalProbe("restore", str(tmp_path), "kv_producer", 0, 4)
    with pytest.raises(RuntimeError, match="snapshot already exists"):
        p.after_forward("req", PROMPT, 0, 19, producer, P_BLOCKS, 4)
