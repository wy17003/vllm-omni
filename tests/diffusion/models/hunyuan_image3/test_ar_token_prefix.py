# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import TokenizerWrapper
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import ImageInfo
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import HunyuanImage3Pipeline
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS
from vllm_omni.diffusion.models.hunyuan_image3.request_layout import (
    extract_hunyuan_ar_token_ids,
    prepare_hunyuan_layout,
)
from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import ar2diffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Tokenizer:
    """Small encoder retaining the two recorded non-invertible token pairs."""

    added_tokens_encoder = {**HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS, "<pad>": 128002, "<timestep>": 128007}
    bos_token_id = added_tokens_encoder["<|startoftext|>"]
    eos_token_id = added_tokens_encoder["<|endoftext|>"]
    pad_token_id = added_tokens_encoder["<pad>"]

    def convert_tokens_to_ids(self, token):
        return self.added_tokens_encoder[token]

    def encode(self, text, add_special_tokens=False):
        if text in self.added_tokens_encoder:
            return [self.added_tokens_encoder[text]]
        if text == "决定了一切":
            return [102111, 100420, 100406]
        if text == "说明了她":
            return [67494, 114745]
        return [ord(char) for char in text]


@pytest.mark.parametrize(
    "text,generated",
    [("决定了一切", [110236, 103550]), ("说明了她", [114980, 100449]), ("stable", list(map(ord, "stable")))],
)
@pytest.mark.parametrize("input_path", ["template", "engine", "worker"])
def test_bridge_to_inputs_preserves_original_tokens(text, generated, input_path, monkeypatch):
    tokenizer = _Tokenizer()
    wrapper = TokenizerWrapper(tokenizer)
    prompt_ids = (
        [tokenizer.bos_token_id]
        + tokenizer.encode("User: draw\n\nAssistant: ")
        + [HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<think>"]]
    )
    generated = generated + [wrapper.end_think_token_id]
    completion = Mock(cumulative_token_ids=generated, cumulative_text=text + "</think>", finish_reason="stop")
    source = Mock(prompt_token_ids=prompt_ids, outputs=[completion], multimodal_output=None)
    bridged = ar2diffusion([source], prompt={"prompt": "draw"})
    raw = extract_hunyuan_ar_token_ids([bridged])
    image_info = ImageInfo(image_type="gen_image", token_width=2, token_height=2, base_size=1024, ratio_index=0)
    arguments = dict(
        batch_prompt=["draw"],
        batch_cot_text=["<think>" + text + "</think>"],
        batch_gen_image_info=[image_info],
        mode="gen_image",
        sequence_template="instruct",
    )
    before = wrapper.apply_chat_template(**arguments)["output"]
    if input_path == "template":
        after = wrapper.apply_chat_template(**arguments, batch_ar_token_ids=raw)["output"]
    else:
        image_processor = Mock(build_image_info=Mock(return_value=image_info))
        generation_config = Mock(sequence_template="instruct", drop_think=False)
        sampling = Mock(height=32, width=32, guidance_scale=0.0, guidance_scale_provided=True, extra_args={})
        request = Mock(prompt=bridged, sampling_params=sampling, request_id="raw-prefix-regression")
        prepared = (
            prepare_hunyuan_layout(
                request,
                tokenizer_wrapper=wrapper,
                image_processor=image_processor,
                generation_config=generation_config,
                image_base_size=1024,
            )
            if input_path == "engine"
            else None
        )
        pipeline = object.__new__(HunyuanImage3Pipeline)
        pipeline._tkwrapper = wrapper
        pipeline.image_processor = image_processor
        pipeline.generation_config = generation_config
        pipeline.config = Mock(image_base_size=1024, attention_head_dim=8, rope_theta=10000.0)
        monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda _self: torch.device("cpu")))
        model_inputs = pipeline.prepare_model_inputs(
            prompt=[bridged["prompt"]],
            cot_text=arguments["batch_cot_text"],
            ar_token_ids=raw,
            mode="gen_image",
            image_size=(32, 32),
            guidance_scale=0.0,
            generator=[torch.Generator().manual_seed(0)],
            prepared_layout=prepared,
        )
        after = model_inputs["tokenizer_output"]
        assert torch.equal(model_inputs["input_ids"], after.tokens)
        assert torch.equal(model_inputs["position_ids"], torch.arange(after.tokens.shape[1])[None])
    expected = prompt_ids + generated
    assert after.tokens[0, : len(expected)].tolist() == expected
    reuse_len = after.think_recaption_end_pos[0][0]
    assert reuse_len == len(expected) - 1
    assert after.tokens[0, reuse_len] == wrapper.end_think_token_id
    old_boundary = before.think_recaption_end_pos[0][0]
    assert torch.equal(before.tokens[:, old_boundary:], after.tokens[:, reuse_len:])
    assert after.gen_image_slices[0][0].start - reuse_len == before.gen_image_slices[0][0].start - old_boundary
    if text == "stable":
        assert torch.equal(before.tokens, after.tokens)
    else:
        assert before.tokens[0, : old_boundary + 1].tolist() != expected

    # Unconditional branches and dropped thought sections keep their existing
    # construction until the corresponding raw-prefix semantics are supported.
    for options in ({"cfg_factor": 2}, {"drop_think": True}):
        old = wrapper.apply_chat_template(**arguments, **options)["output"]
        new = wrapper.apply_chat_template(**arguments, **options, batch_ar_token_ids=raw)["output"]
        assert torch.equal(old.tokens, new.tokens)


@pytest.mark.parametrize("finish_reason,terminal", [("length", "</think>"), ("stop", "</recaption>")])
def test_other_stop_paths_keep_text_template(finish_reason, terminal):
    prompt = {
        "extra": {
            "ar_prompt_token_ids": [127958, 128023],
            "ar_token_ids": [100, HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS[terminal]],
            "ar_finish_reason": finish_reason,
        }
    }
    assert extract_hunyuan_ar_token_ids([prompt]) == [None]


@pytest.mark.parametrize("lengths", [((3, 3), (3, 3)), ((4, 4), (4, 3)), ((4, 4), (5, 5))])
def test_invalid_kv_is_rejected_before_any_injection(lengths):
    layers = [Mock(layer_idx=i) for i in range(2)]
    for layer in layers:
        layer.self_attn.image_attn._injected_ar_kv = None
    pipeline = Mock(model=Mock(layers=layers))
    kv = {i: {"key": torch.zeros(k, 1, 1), "value": torch.zeros(v, 1, 1)} for i, (k, v) in enumerate(lengths)}
    with pytest.raises(ValueError, match="AR KV does not cover"):
        HunyuanImage3Pipeline.inject_ar_kv_into_layers(pipeline, kv, 4)
    assert all(layer.self_attn.image_attn._injected_ar_kv is None for layer in layers)


def test_transfer_metadata_matches_actual_kv():
    sampling = Mock(
        past_key_values=Mock(key_cache=[torch.zeros(4, 1, 1)], value_cache=[torch.zeros(4, 1, 1)]),
        kv_metadata={"seq_len": 5},
    )
    with pytest.raises(ValueError, match="transferred seq_len=5"):
        HunyuanImage3Pipeline._extract_ar_kv_from_sampling(None, sampling)
