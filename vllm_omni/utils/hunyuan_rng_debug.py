# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only first-two-sample diagnostics for the Hunyuan PD RNG experiment."""

from __future__ import annotations

import json
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.utils.debug_fingerprint import bytes_fingerprint, int_sequence_fingerprint, tensor_fingerprint

logger = init_logger(__name__)


def _generator_fingerprint(generator: torch.Generator | None) -> dict[str, Any]:
    if generator is None:
        return {"error": "No request-level generator"}
    try:
        state = generator.get_state().detach().cpu().contiguous()
        result = {
            "initial_seed": int(generator.initial_seed()),
            "device": str(generator.device),
            "state_numel": state.numel(),
            "state_sha256": bytes_fingerprint(state.view(torch.uint8).numpy().tobytes()),
        }
        try:
            result["offset"] = int(generator.get_offset())
        except (RuntimeError, NotImplementedError):
            pass  # CPU generators, for example, do not expose an offset.
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _synchronize(device: torch.device) -> None:
    if device.type != "cpu":
        # The NPU sampler may use another stream. Diagnostics must observe all
        # submitted work, without drawing random numbers or changing RNG state.
        torch.get_device_module(device).synchronize(device)


def sample_with_rng_debug(sampler, logits, sampling_metadata, request_ids, role, tp_rank):
    """Observe RNG without get/set round-trips, replay, or additional sampling.

    The temporary hook runs at the top-k/top-p sampler entry, after the generic
    sampler's penalties and temperature processing but BEFORE top-k/top-p.
    In the supplied experiment top_k is disabled and top_p=1, so neither filter
    changes this distribution. Model sampler input is recorded separately.
    """
    rows = {}
    for index in range(logits.shape[0]):
        history = sampling_metadata.output_token_ids[index]
        if len(history) < 2:
            rows[index] = {
                "request_id": request_ids[index] if index < len(request_ids) else None,
                "role": role,
                "tp_rank": tp_rank,
                "output_ordinal": len(history) + 1,
                "history_sha256": int_sequence_fingerprint(history),
                "sampler_class": f"{type(sampler).__module__}.{type(sampler).__name__}",
                "experiment": "rng_without_handoff",
                "random_sampler_called": False,
            }
    if not rows:
        return sampler(logits=logits, sampling_metadata=sampling_metadata)

    def capture(callback, row, field):
        try:
            row[field] = callback()
        except Exception as exc:
            row[field] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        _synchronize(logits.device)
    except Exception as exc:
        for row in rows.values():
            row["synchronize_before_error"] = str(exc)
    for index, row in rows.items():
        row["rng_before"] = _generator_fingerprint(sampling_metadata.generators.get(index))
        capture(lambda: tensor_fingerprint(logits[index]), row, "model_sampler_input_logits")
        controls = {}
        for name in (
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalties",
            "presence_penalties",
            "frequency_penalties",
        ):
            value = getattr(sampling_metadata, name, None)
            capture(lambda: value[index].item() if value is not None else None, controls, name)
        row["sampling_controls"] = controls

    def before_random_sample(module, args):
        # This is a read-only nn.Module pre-hook; returning None leaves args intact.
        for index, row in rows.items():
            row["random_sampler_called"] = True
            row["random_sampler_class"] = f"{type(module).__module__}.{type(module).__name__}"
            capture(lambda: tensor_fingerprint(args[0][index]), row, "random_sampler_input_logits")
            capture(lambda: _generator_fingerprint(args[1].get(index)), row, "rng_at_random_sample")

    handle = None
    try:
        random_sampler = getattr(sampler, "topk_topp_sampler", None)
        if random_sampler is not None:
            try:
                handle = random_sampler.register_forward_pre_hook(before_random_sample)
            except Exception as exc:
                for row in rows.values():
                    row["random_sampler_hook_error"] = str(exc)
        result = sampler(logits=logits, sampling_metadata=sampling_metadata)
    finally:
        if handle is not None:
            handle.remove()

    try:
        _synchronize(logits.device)
    except Exception as exc:
        for row in rows.values():
            row["synchronize_after_error"] = str(exc)
    for index, row in rows.items():
        row["rng_after"] = _generator_fingerprint(sampling_metadata.generators.get(index))
        capture(lambda: result.sampled_token_ids[index].detach().cpu().tolist(), row, "sampled_token_ids")
        # This is immediately after sampling, BEFORE worker bookkeeping can
        # discard a partial-prefill sample and roll back its RNG offset.
        row["event"] = "sample_before_bookkeeping"
        logger.info("[HY3_AR_RNG] %s", json.dumps(row, sort_keys=True, separators=(",", ":")))
    return result
