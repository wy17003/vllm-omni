# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_request() -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[{"prompt": "a cup of coffee on a table"}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id="request-test",
    )


def test_request_id_is_required():
    with pytest.raises(TypeError, match="request_id"):
        OmniDiffusionRequest(
            prompts=[{"prompt": "a cup of coffee on a table"}],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        )


def test_request_ids_identity_list_is_removed():
    req = _make_request()

    assert req.request_id == "request-test"
    assert not hasattr(req, "request_ids")


def test_request_id_must_be_non_empty():
    with pytest.raises(ValueError, match="request_id must be a non-empty string"):
        OmniDiffusionRequest(
            prompts=[{"prompt": "a cup of coffee on a table"}],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
            request_id="",
        )


def test_dummy_run_request_is_identified_by_reserved_request_id():
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "dummy run"}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
    )

    assert req.is_dummy_run()
    assert OmniDiffusionRequest.is_dummy_run_request_id(DUMMY_DIFFUSION_REQUEST_ID)


def test_non_dummy_request_is_not_identified_as_dummy_run():
    req = _make_request()

    assert not req.is_dummy_run()
    assert not OmniDiffusionRequest.is_dummy_run_request_id(req.request_id)
    assert not OmniDiffusionRequest.is_dummy_run_request_id(None)


def test_tp_seed_same_across_ranks_and_varies_across_requests():
    random.seed(0)
    n_requests = 5
    seeds = [_make_request().sampling_params.seed for _ in range(n_requests)]

    # Seed must be auto-assigned (not None) so every TP rank can use it.
    assert all(s is not None for s in seeds)

    # Seeds must vary across requests (non-determinism preserved).
    assert len(set(seeds)) == n_requests, f"Expected {n_requests} unique seeds but got {len(set(seeds))}: {seeds}"


@pytest.mark.parametrize(
    "scale,provided,expected_scale,expected_provided",
    [(0.0, True, 0.0, True), (0.0, False, 1.0, False), (5.0, False, 5.0, True), (1.0, False, 1.0, True)],
)
def test_request_initialization_preserves_explicit_zero(scale, provided, expected_scale, expected_provided):
    params = OmniDiffusionSamplingParams(seed=42, guidance_scale=scale, guidance_scale_provided=provided)
    req = OmniDiffusionRequest(request_id="request", prompts=["cat"], sampling_params=params)
    assert params.guidance_scale == expected_scale
    assert params.guidance_scale_provided is expected_provided
    assert params.guidance_scale_2 == expected_scale
    if provided:
        req.__post_init__()
        assert params.guidance_scale == scale
