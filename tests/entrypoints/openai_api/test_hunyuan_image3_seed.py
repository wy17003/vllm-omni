# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HunyuanImage3 request seed overrides through real pipeline metadata."""

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    return object.__new__(OmniOpenAIServingChat)


@pytest.mark.parametrize(
    "pipeline_name", ["HUNYUAN_IMAGE3_PIPELINE", "HUNYUAN_IMAGE3_AR_PIPELINE", "HUNYUAN_IMAGE3_PD_PIPELINE"]
)
@pytest.mark.parametrize("request_seed", [43, 0, None])
def test_hunyuan_request_seed_uses_real_pipeline_ownership(serving_chat, pipeline_name, request_seed):
    from vllm_omni.config.omni_config import BaseVllmOmniStageConfig
    from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin
    from vllm_omni.model_executor.models.hunyuan_image3 import pipeline as hy3_pipeline

    serving_chat._diffusion_extra_body_params = frozenset()
    topology = getattr(hy3_pipeline, pipeline_name)
    stages = []
    defaults = []
    for stage in topology.stages:
        # Exercise the real structured-config properties and real topology.
        # No model/runtime is needed to project stage_type/is_comprehension.
        config = object.__new__(BaseVllmOmniStageConfig)
        config.stage_pipeline_config = stage
        stages.append(config)
        defaults.append(
            SamplingParams(temperature=1, seed=42, max_tokens=32)
            if config.stage_type == "llm"
            else OmniDiffusionSamplingParams(seed=42)
        )
    engine = SimpleNamespace(
        stage_configs=stages,
        default_sampling_params_list=defaults,
        _pd_separation_pair=PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages),
    )
    _, resolved = serving_chat._build_multistage_generation_inputs(
        engine=engine,
        prompt="generate a sleeping cat",
        extra_body={},
        reference_images=[],
        gen_params=OmniDiffusionSamplingParams(seed=request_seed),
    )
    expected_seed = 42 if request_seed is None else request_seed
    for stage, params in zip(stages, resolved):
        if stage.stage_type == "llm":
            assert params.seed == expected_seed
            assert params.temperature == 1 and params.max_tokens == 32
    assert all(params.seed == 42 for params in defaults)


@pytest.mark.parametrize("pd", [False, True])
@pytest.mark.parametrize("request_scale", [None, 0.0, 1.0, 5.0])
@pytest.mark.parametrize("default_scale", [None, 0.0, 5.0])
def test_hunyuan_guidance_survives_defaults_overrides_and_request_normalization(
    serving_chat, pd, request_scale, default_scale
):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.engine.stage_init_utils import extract_stage_metadata

    serving_chat._diffusion_extra_body_params = frozenset()
    stages = [SimpleNamespace(stage_type="llm", is_comprehension=True)]
    defaults = [SamplingParams(seed=42)]
    if pd:
        stages.append(SimpleNamespace(stage_type="llm", is_comprehension=False))
        defaults.append(SamplingParams(seed=42))
    yaml_params = {} if default_scale is None else {"guidance_scale": default_scale}
    config = SimpleNamespace(
        stage_id=len(stages), stage_type="diffusion", engine_args={}, default_sampling_params=yaml_params
    )
    stages.append(config)
    defaults.append(extract_stage_metadata(config).default_sampling_params)
    engine = SimpleNamespace(
        stage_configs=stages, default_sampling_params_list=defaults, _pd_separation_pair=(0, 1) if pd else None
    )
    extra_body = {} if request_scale is None else {"guidance_scale": request_scale}
    _, resolved = serving_chat._build_multistage_generation_inputs(
        engine=engine,
        prompt="cat",
        extra_body=extra_body,
        reference_images=[],
        gen_params=OmniDiffusionSamplingParams(seed=42),
    )
    params = resolved[-1]
    OmniDiffusionRequest(request_id="guidance-test", prompts=["cat"], sampling_params=params)
    expected = request_scale if request_scale is not None else default_scale
    assert params.guidance_scale_provided is (expected is not None)
    assert params.guidance_scale == (1.0 if expected is None else expected)
    assert defaults[-1].guidance_scale == (0.0 if default_scale is None else default_scale)
    assert defaults[-1].guidance_scale_provided is (default_scale is not None)
    assert yaml_params == ({} if default_scale is None else {"guidance_scale": default_scale})


@pytest.mark.parametrize("scale", [None, 0.0, 1.0, 5.0])
def test_images_parameter_update_marks_explicit_guidance(scale):
    from vllm_omni.entrypoints.openai.api_server import _update_if_not_none

    params = OmniDiffusionSamplingParams()
    _update_if_not_none(params, "guidance_scale", scale)
    assert params.guidance_scale_provided is (scale is not None)
    assert params.guidance_scale == (0.0 if scale is None else scale)
