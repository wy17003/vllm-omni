# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.utils.debug_fingerprint import tensor_fingerprint
from vllm_omni.utils.hunyuan_rng_debug import _generator_fingerprint, sample_with_rng_debug

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RandomSampler(torch.nn.Module):
    def forward(self, logits, generators, top_k, top_p):
        probabilities = logits.softmax(dim=-1)
        noise = torch.empty_like(probabilities)
        for index, generator in generators.items():
            noise[index].exponential_(generator=generator)
        return (probabilities / noise).argmax(dim=-1)


class _Sampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = _RandomSampler()

    def forward(self, logits, sampling_metadata):
        transformed = logits.float() / sampling_metadata.temperature[:, None]
        tokens = self.topk_topp_sampler(
            transformed, sampling_metadata.generators, sampling_metadata.top_k, sampling_metadata.top_p
        )
        return SimpleNamespace(sampled_token_ids=tokens[:, None])


def _metadata(generator, history):
    return SimpleNamespace(
        generators={0: generator},
        output_token_ids=[history],
        temperature=torch.tensor([1.0]),
        top_k=None,
        top_p=None,
    )


def _rows(caplog):
    return [json.loads(r.message.split("[HY3_AR_RNG] ", 1)[1]) for r in caplog.records if "[HY3_AR_RNG] " in r.message]


def test_generator_fingerprint_does_not_advance_state():
    generator = torch.Generator().manual_seed(42)
    before = generator.get_state().clone()
    fingerprint = _generator_fingerprint(generator)
    assert fingerprint["initial_seed"] == 42
    assert fingerprint["state_numel"] == before.numel()
    assert torch.equal(before, generator.get_state())
    assert fingerprint == _generator_fingerprint(generator)


def test_probe_preserves_samples_rng_and_logits_and_captures_transformed_input(caplog):
    caplog.set_level(logging.INFO)
    observed = torch.Generator().manual_seed(42)
    baseline = torch.Generator().manual_seed(42)
    sampler = _Sampler()
    logits = torch.arange(16, dtype=torch.bfloat16).reshape(1, -1) / 10
    original = logits.clone()
    history = []
    for ordinal in (1, 2):
        metadata = _metadata(observed, history)
        metadata.temperature.fill_(0.5)
        reference = _metadata(baseline, history)
        reference.temperature.fill_(0.5)
        expected = sampler(logits, reference)
        actual = sample_with_rng_debug(sampler, logits, metadata, ["req"], "non_pd", 0)
        assert torch.equal(expected.sampled_token_ids, actual.sampled_token_ids)
        assert torch.equal(observed.get_state(), baseline.get_state())
        assert torch.equal(logits, original)
        assert not sampler.topk_topp_sampler._forward_pre_hooks
        history.append(actual.sampled_token_ids.item())
        row = _rows(caplog)[-1]
        assert row["output_ordinal"] == ordinal and row["random_sampler_called"]
        assert row["rng_before"]["state_sha256"] == row["rng_at_random_sample"]["state_sha256"]
        assert row["rng_after"]["state_sha256"] != row["rng_before"]["state_sha256"]
        assert row["random_sampler_input_logits"] == tensor_fingerprint(logits[0].float() / 0.5)
    assert len(_rows(caplog)) == 2


def test_probe_observes_consumer_rng_reset_without_repairing_it(caplog):
    caplog.set_level(logging.INFO)
    sampler = _Sampler()
    logits = torch.zeros(1, 32)
    producer = torch.Generator().manual_seed(42)
    first = sample_with_rng_debug(sampler, logits, _metadata(producer, []), ["req"], "kv_producer", 0)
    consumer = torch.Generator().manual_seed(42)
    sample_with_rng_debug(
        sampler, logits, _metadata(consumer, first.sampled_token_ids[0].tolist()), ["req"], "kv_consumer", 0
    )
    p, d = _rows(caplog)
    assert d["output_ordinal"] == 2
    assert d["rng_before"]["state_sha256"] == p["rng_before"]["state_sha256"]
    assert d["rng_before"]["state_sha256"] != p["rng_after"]["state_sha256"]


def test_probe_is_skipped_after_second_output(caplog):
    sampler = Mock(return_value=object())
    metadata = _metadata(torch.Generator(), [1, 2])
    logits = torch.zeros(1, 4)
    assert sample_with_rng_debug(sampler, logits, metadata, ["req"], "non_pd", 0) is sampler.return_value
    sampler.assert_called_once_with(logits=logits, sampling_metadata=metadata)
    assert not _rows(caplog)


def test_probe_removes_hook_on_sampler_failure():
    sampler = _Sampler()
    sampler.topk_topp_sampler.forward = Mock(side_effect=RuntimeError("sampling failed"))
    with pytest.raises(RuntimeError, match="sampling failed"):
        sample_with_rng_debug(sampler, torch.zeros(1, 4), _metadata(torch.Generator(), []), ["req"], "non_pd", 0)
    assert not sampler.topk_topp_sampler._forward_pre_hooks


def test_generator_export_failure_is_explicit():
    generator = Mock()
    generator.get_state.side_effect = RuntimeError("unsupported backend")
    assert "unsupported backend" in _generator_fingerprint(generator)["error"]
    assert "error" in _generator_fingerprint(None)
