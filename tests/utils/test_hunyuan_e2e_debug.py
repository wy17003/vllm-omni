# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import hashlib
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

from vllm_omni.utils.debug_fingerprint import full_kv_fingerprint, full_tensor_digest
from vllm_omni.utils.hunyuan_e2e_debug import HunyuanE2EProbe

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "vllm_omni/diffusion/models/hunyuan_image3"


def _events(caplog):
    return [
        (record.message.split("event=", 1)[1].split()[0], json.loads(record.message.split(" data=", 1)[1]))
        for record in caplog.records
        if "[HY3_E2E]" in record.message and " data=" in record.message
    ]


@pytest.mark.parametrize(
    "tensor",
    [torch.arange(30).reshape(5, 6).T, torch.ones(13, dtype=torch.bfloat16), torch.tensor(1.0), torch.empty(0)],
)
def test_full_digest_hashes_logical_bytes_across_chunks(tensor):
    expected = hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    assert full_tensor_digest(tensor, chunk_bytes=7)["sha256"] == expected


def test_full_kv_detects_unsampled_layer_and_tail_and_rejects_missing():
    keys = [torch.zeros(1024) for _ in range(8)]
    values = [key.clone() for key in keys]
    before = full_kv_fingerprint(keys, values)
    values[2][501] = 1
    after = full_kv_fingerprint(keys, values)
    assert before["sha256"] != after["sha256"]
    assert before["layers"][5]["sha256"] != after["layers"][5]["sha256"]
    with pytest.raises(ValueError):
        full_kv_fingerprint(keys, values[:-1])
    with pytest.raises(ValueError):
        full_kv_fingerprint([], [])


def test_probe_reads_injected_kv_without_clearing_and_saves_exact_latents(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    key = torch.arange(16, dtype=torch.bfloat16).reshape(4, 2, 2)
    entries = [(key, key.clone())]
    layer = SimpleNamespace(self_attn=SimpleNamespace(image_attn=SimpleNamespace(_injected_ar_kv=entries)))
    probe = HunyuanE2EProbe("req/test", 3, {"debug_e2e_dump_dir": str(tmp_path)})
    before = torch.random.get_rng_state().clone()
    probe.injected_kv([layer])
    probe.record("dit_final", tensors={"latents": key})
    assert layer.self_attn.image_attn._injected_ar_kv is entries
    assert entries[0][0] is key
    assert torch.equal(before, torch.random.get_rng_state())
    saved = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert torch.equal(saved["latents"], key)
    assert [event for event, _ in _events(caplog)] == ["dit_injected_kv", "dit_final", "artifact"]
    probe.injected_kv([SimpleNamespace()])
    assert "event=probe_error" in caplog.text


def _method(path, cls_name, method, namespace):
    """Execute the real method body with CPU fakes, without loading NPU/model dependencies."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if cls_name is None:
        nodes = ast.walk(tree)
    else:
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls_name)
        nodes = cls.body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == method)
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method]


@pytest.mark.parametrize(
    "prompt,explicit,expected",
    [
        ({"prompt": "cat", "height": 832, "width": 1216}, (None, None), (832, 1216)),
        ({"prompt": "cat", "height": 832, "width": 1216}, (512, 768), (512, 768)),
        ("cat", (None, None), (None, None)),
        ({"prompt": "cat", "multi_modal_data": {"image": "image"}}, (None, None), (600, 800)),
        (
            {"prompt": "cat", "height": 832, "width": 1216, "multi_modal_data": {"image": "image"}},
            (None, None),
            (832, 1216),
        ),
    ],
)
def test_preprocess_transfers_bridge_size_for_text_and_image_inputs(prompt, explicit, expected):
    preprocess = _method(
        MODEL_DIR / "pipeline_hunyuan_image3.py",
        None,
        "pre_process_func",
        {
            "OmniTextPrompt": dict,
            "_build_cond_joint_image": lambda image: {"fake": image},
            "_to_pil_image": lambda image: SimpleNamespace(size=(800, 600)),
        },
    )
    params = SimpleNamespace(height=explicit[0], width=explicit[1])
    req = SimpleNamespace(prompts=[prompt], sampling_params=params)
    assert preprocess(req) is req
    assert (params.height, params.width) == expected


def test_probe_reports_cfg_branches_separately_without_mutation(caplog):
    caplog.set_level(logging.INFO)
    key = torch.arange(12).reshape(3, 2, 2)
    entries = [(key, key.clone()), (key + 10, key + 20)]
    layer = SimpleNamespace(self_attn=SimpleNamespace(image_attn=SimpleNamespace(_injected_ar_kv=entries)))
    probe = HunyuanE2EProbe("cfg-request", 0, {})
    probe.injected_kv([layer], branch_roles=("positive", "negative"))
    rows = _events(caplog)
    assert [data["branch"] for _, data in rows] == ["positive", "negative"]
    assert rows[0][1]["kv"]["sha256"] != rows[1][1]["kv"]["sha256"]
    assert layer.self_attn.image_attn._injected_ar_kv is entries
    assert "probe_error" not in caplog.text
    probe.injected_kv([layer])
    assert "expected KV branches ('positive',), got 2" in caplog.text


def test_ordinary_forward_reaches_denoise_probes_without_changing_output_or_rng(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    namespace = {
        "torch": torch,
        "PipelineCallback": type("PipelineCallback", (), {}),
        "MultiPipelineCallbacks": type("MultiPipelineCallbacks", (), {}),
        "retrieve_timesteps": lambda scheduler, count, *args: (torch.arange(count, 0, -1), count),
        "set_forward_context_denoise_step_idx": lambda step: None,
        "HunyuanImage3Text2ImagePipelineOutput": lambda samples: (samples,),
        "get_tensor_model_parallel_world_size": lambda: 4,
        "get_sequence_parallel_world_size": lambda: 1,
        "get_classifier_free_guidance_world_size": lambda: 1,
    }
    call = _method(
        MODEL_DIR / "hunyuan_image3_transformer.py", "HunyuanImage3Text2ImagePipeline", "__call__", namespace
    )
    generate = _method(MODEL_DIR / "pipeline_hunyuan_image3.py", "HunyuanImage3Pipeline", "_generate", {})
    forward = _method(
        MODEL_DIR / "pipeline_hunyuan_image3.py",
        "HunyuanImage3Pipeline",
        "forward",
        {"logger": logging.getLogger(__name__), "DiffusionOutput": SimpleNamespace},
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.distributed.omni_connectors.utils.kv_utils",
        SimpleNamespace(get_local_tp_rank=lambda: 0),
    )
    tensor = torch.zeros(1, 1, 2, 2)
    entries = [(tensor, tensor.clone())]
    layer = SimpleNamespace(self_attn=SimpleNamespace(image_attn=SimpleNamespace(_injected_ar_kv=entries)))
    mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
    model = SimpleNamespace(
        config=SimpleNamespace(vae={"latent_channels": 1}),
        generation_config=None,
        model=SimpleNamespace(layers=[layer]),
        _prepare_attention_mask_for_generation=lambda *args, **kwargs: mask,
        prepare_inputs_for_generation=lambda ids, **kwargs: kwargs,
        forward_call=lambda images, **kwargs: {"diffusion_prediction": images.float() * 0.25},
        _update_model_kwargs_for_generation=lambda output, kwargs: kwargs,
    )

    class FakeImage:
        size = (2, 2)
        mode = "L"

        def __init__(self, pixels):
            self.pixels = pixels

        def tobytes(self):
            return self.pixels.numpy().tobytes()

    pipeline = SimpleNamespace(
        model=model,
        do_classifier_free_guidance=False,
        device=torch.device("cpu"),
        _execution_device=torch.device("cpu"),
        scheduler=SimpleNamespace(order=1, config={}, step=lambda pred, t, x, **kwargs: (x - pred,)),
        vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=0.5), decode=lambda x, **kwargs: (x,)),
        image_processor=SimpleNamespace(postprocess=lambda x, **kwargs: [FakeImage(x)]),
        prepare_latents=lambda generator, **kwargs: torch.randn(1, 1, 2, 2, generator=generator),
        prepare_extra_func_kwargs=lambda *args: {},
        _maybe_handle_ar_kv_reuse=lambda ids, *args: (ids, 3),
        progress_bar=lambda **kwargs: nullcontext(SimpleNamespace(update=lambda: None)),
    )
    seen = []

    def invoke(**kwargs):
        assert "_hy3_e2e_probe" not in kwargs["model_kwargs"]
        seen.append(kwargs.get("debug_probe"))
        return call(pipeline, **kwargs)

    wrapper = SimpleNamespace(pipeline=invoke)
    info = SimpleNamespace(
        image_token_length=4, add_timestep_token=False, add_guidance_token=False, image_height=2, image_width=2
    )
    wrapper._extract_prompt_inputs = lambda *args, **kwargs: (["cat"], ["CoT"], "system", None, "think")
    wrapper._normalize_cot_text = lambda text: text
    wrapper._extract_ar_kv_from_request = lambda req: {"ar_kv_data": {0: {"key": tensor, "value": tensor}}}
    wrapper.prepare_model_inputs = lambda **kwargs: {
        **kwargs,
        "batch_gen_image_info": [info],
        "input_ids": torch.tensor([[1, 2, 3]]),
    }
    wrapper._generate = lambda **kwargs: generate(wrapper, **kwargs)

    def run(enabled):
        generator = torch.Generator().manual_seed(43)
        wrapper.od_config = SimpleNamespace(omni_kv_config={"debug_e2e": enabled}, step_execution=False)
        req = SimpleNamespace(
            request_id="request",
            prompts=["cat"],
            sampling_params=SimpleNamespace(
                generator=generator,
                seed=43,
                height=2,
                width=2,
                num_inference_steps=12,
                guidance_scale_provided=True,
                guidance_scale=0,
            ),
        )
        output = forward(wrapper, req)
        return output.output.tobytes(), generator.get_state()

    baseline, baseline_rng = run(False)
    assert _events(caplog) == []
    observed, observed_rng = run(True)
    assert observed == baseline
    assert torch.equal(observed_rng, baseline_rng)
    assert seen[0] is None and isinstance(seen[1], HunyuanE2EProbe)
    events = _events(caplog)
    assert [event for event, _ in events] == [
        "dit_request",
        "dit_initial",
        "dit_condition",
        "dit_injected_kv",
        "dit_step",
        "dit_step",
        "dit_step",
        "dit_final",
        "vae_output",
        "image",
    ]
    assert [data["step"] for event, data in events if event == "dit_step"] == [1, 10, 12]
    assert "event=probe_error" not in caplog.text
    assert layer.self_attn.image_attn._injected_ar_kv is entries


def test_e2e_yaml_resolved_sampling_and_npu_layout():
    # Use the real resolver functions; avoid importing unrelated model registries.
    path = ROOT / "vllm_omni/config/stage_config.py"
    names = {
        "_deep_merge_stage",
        "_get_recursively_merged_dict",
        "_merge_stage_lists",
        "_merge_platforms",
        "resolve_deploy_yaml",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "Path": Path,
        "Any": Any,
        "logger": logging.getLogger(__name__),
        "_DEEP_MERGE_KEYS": {"default_sampling_params", "subtalker_sampling_params", "engine_extras", "engine_args"},
        "load_yaml_config": lambda p: yaml.safe_load(Path(p).read_text()),
        "to_dict": lambda value: value,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    resolve = namespace["resolve_deploy_yaml"]
    for suffix, temperature in [("e2e_check", 0), ("e2e_rng_check", 1)]:
        non_pd, pd = [
            resolve(ROOT / f"vllm_omni/deploy/hunyuan_image_3_moe{mode}_{suffix}.yaml") for mode in ("", "_pd")
        ]
        ar, p, d = non_pd["stages"][0], pd["stages"][0], pd["stages"][1]
        params = dict(d["default_sampling_params"])
        assert params.pop("extra_args") == {"pd_resume_from_prefill": True, "pd_rng_repro_without_state": False}
        assert params == ar["default_sampling_params"]
        assert params["temperature"] == temperature and params["max_tokens"] == 8192
        assert params["seed"] == 42 and not params["ignore_eos"]
        assert p["async_scheduling"] is False and "default_sampling_params" not in p
        assert ar["async_scheduling"] is d["async_scheduling"] is True
        assert p["env"]["VLLM_MOONCAKE_BOOTSTRAP_PORT"] == "25201"
        for config in (non_pd, pd):
            stages = config["stages"]
            dit_params = stages[-1]["default_sampling_params"]
            post_init = _method(ROOT / "vllm_omni/diffusion/request.py", "OmniDiffusionRequest", "__post_init__", {})
            sampling = SimpleNamespace(**dit_params, generator=None, guidance_scale_2=None)
            post_init(SimpleNamespace(request_id="yaml-test", prompts=["cat"], sampling_params=sampling))
            assert sampling.guidance_scale == 0 and sampling.guidance_scale_provided is True
            assert stages[-1]["step_execution"] is False
            assert stages[-1]["omni_kv_config"]["need_recv_cache"] is True
            assert stages[-1]["omni_kv_config"]["debug_e2e"] is True
            assert stages[-2]["omni_kv_config"]["need_send_cache"] is True
            assert config["platforms"]["npu"]["stages"][-2]["devices"] == "8,9,10,11"
            assert config["platforms"]["npu"]["stages"][-1]["devices"] == "12,13,14,15"
        assert non_pd["stages"][-1]["default_sampling_params"] == pd["stages"][-1]["default_sampling_params"]
