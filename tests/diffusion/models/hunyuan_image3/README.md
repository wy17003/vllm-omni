# HunyuanImage3 AR KV prefix regression

The first fix preserves the original AR prompt and generated token IDs for
completed `think` outputs in text-to-image generation with CFG disabled and
`drop_think=False`. The existing template appends the image/control suffix and
builds its positions and masks. The terminal `</think>` remains in the suffix
computed by DiT. KV tensor lengths must cover the reuse boundary and agree with
the transferred `seq_len` when that metadata is present.

The bridge carries the raw prompt IDs, generated IDs, and finish reason. Template
selection uses the actual prompt trigger and terminal token IDs, so special-token
removal from the decoded output cannot discard these boundaries. Other stop
paths, conditional images, and CFG branches retain their existing construction.

## CPU regression

In a configured vLLM-Omni development environment:

```bash
pytest -q tests/diffusion/models/hunyuan_image3/test_ar_token_prefix.py \
  tests/model_executor/stage_input_processors/test_hunyuan_image3_bridge.py
```

The tests cover both recorded non-invertible token pairs, a stable encoding,
suffix preservation, CFG/drop-think compatibility, other stop paths, and invalid
KV lengths. Each token case runs through the template directly, Engine
`prepare_hunyuan_layout` followed by Worker `prepare_model_inputs`, and Worker
local `prepare_model_inputs`. This catches missing raw-token forwarding in
either preparation path. The tests do not require model weights.

Local replay on 2026-10-22 used the original full AR token sequences and tokenizer,
CPU PyTorch, and source-isolated production bridge, layout, template, RoPE,
mask, truncation, and cache-concatenation functions. It checked CPU attention
against the directly concatenated original KV and compared Engine and Worker
inputs. This is CPU simulation, not a full vLLM server or device inference run.

| Case | Generated / re-encoded tokens | Reused KV | New queries | Full sequence |
| --- | --- | --- | --- | --- |
| Original NPD failure | 466 / 467 | 1777 | 4105 | 5882 |
| PD with different IDs at equal length | 458 / 458 | 2561 | 4105 | 6666 |

The main-branch baseline `c04bef8e5fd5b47e5559384a2fb0907a79574d36`
reproduced `q_len(4105) + ar_kv_len(1777) != seq_len(5883)` on the same CPU replay.
The PD token fixture was replayed at 1024 x 1024 to exercise token identity;
these numbers are not a reproduction of its original concurrent workload.

## Remaining acceptance plan

1. **Retain the CPU regressions.** Keep the length-changing, equal-length but
   different-ID, and stable-encoding cases, including suffix and KV-boundary
   checks. The original PD/NPD samples serve as token regression data; reproducing
   their deployment topology or concurrent workload is not an acceptance gate.
2. **Run one GPU functional acceptance.** Use a main-based checkout containing
   this fix, with T2I, fixed dimensions, `think`, CFG disabled, and legacy AR KV
   import. Run one sample whose actual generated IDs change under decode/encode
   and one stable-encoding control through the normal complete denoising flow.
   Verify the token property from the actual AR output, rather than assuming the
   same prompt or seed guarantees it. Record the base/fix revisions, GPU, model
   and request settings.
3. **Prove that AR KV was reused.** For each GPU sample, verify that the DiT prefix
   matches the original AR IDs, the transferred KV covers the reuse boundary,
   and `</think>` is the first uncached suffix token. Record a nonzero injected
   prefix length and the first-step query length, with
   `reused_prefix_length + query_length == full_sequence_length`, and confirm
   that full-prefix recomputation was not used. A successful image alone does
   not establish that the corrected reuse path ran.

GPU acceptance is pending. Once these checks pass, the conclusion is limited to:
**CPU regressions pass, and the target AR KV reuse path completes real inference
in a main-based GPU environment.** This does not establish other task modes,
transport modes, image-quality improvements, or performance gains.
