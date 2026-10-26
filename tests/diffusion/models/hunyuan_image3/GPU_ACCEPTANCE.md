# GPU acceptance with the original request

This procedure expands “Run one GPU functional acceptance” and “Prove that AR
KV was reused” in [README.md](README.md). CPU regression has passed. **The GPU
commands below have not been executed locally.** Use the main-based fix
`a7bb6e891` (base `c04bef8e5`), or record the actual newer revision containing it.

## Prepare the original inputs and GPU configuration

Run from the repository root in a configured Linux/CUDA vLLM-Omni environment,
using eight GPUs with enough memory for the original checkpoint and TP4 per
stage. Keep the model at `/new_data/HunyuanImage-3.0-Instruct/`. Copy the original
`D:\Data\pdd\kv_reuse\kv_reuse` evidence bundle to
`/home/w00934428/kv_reuse` on that machine. The commands below use Bash.

The original launch file is `evidence/start_original.txt`; the actual deployment
is `evidence/npd/declared_config.yaml`. Its device numbers refer to NPU cards.
Preserve its sampling, TP4, shared-memory transport, eager execution,
`step_execution: false`, and `TORCH_SDPA`. Make these explicit GPU adaptations:

| Setting | Original | GPU acceptance |
| --- | --- | --- |
| AR devices | `4,5,6,7` | `0,1,2,3` |
| DiT devices | `8,9,10,11` | `4,5,6,7` |
| AR prefix caching | Enabled globally | Still enabled |
| DiT prefix caching | Inherited global setting | Explicitly disabled |
| DiT KV mode | Implicit legacy | Explicit `dense_legacy` |

The DiT override is required by current main: scheduler prefix caching requires
`paged_scheduler`, which is outside this acceptance. It does **not** disable
legacy AR KV import (`need_send_cache` / `need_recv_cache` remain enabled).

```bash
export HY3_BUNDLE=/home/w00934428/kv_reuse
export HY3_RUN=/home/w00934428/hy3_run01
mkdir -p "$HY3_RUN/gpu_acceptance" experiments/hunyuan_pd/configs

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import yaml

bundle = Path(os.environ["HY3_BUNDLE"])
out = Path(os.environ["HY3_RUN"]) / "gpu_acceptance"
cfg = yaml.safe_load((bundle / "evidence/npd/declared_config.yaml").read_text(encoding="utf-8"))
ar, dit = cfg["stages"]
ar["devices"], dit["devices"] = "0,1,2,3", "4,5,6,7"
dit["enable_prefix_caching"] = False
dit["diffusion_kv_mode"] = "dense_legacy"
Path("experiments/hunyuan_pd/configs/npd.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

# Preserve the failing request byte for byte.
original = (bundle / "evidence/request_0007.json").read_bytes()
(out / "request_0007.json").write_bytes(original)
# Original round-1 successful request 0001 is a stable-control candidate.
records = json.loads((bundle / "evidence/npd/round1.json").read_text(encoding="utf-8"))["records"]
record = next(r for r in records if r["row"] == 1)
assert record["success"] and record["id"] == "scene20_short"
saved = Path(record["request_file"])
if saved.exists():
    (out / "request_0001.json").write_bytes(saved.read_bytes())
else:
    # The bundle omits 0001's HTTP body; recover it from the archived request.
    rows = [json.loads(s) for s in (bundle / "reference_configs/repository_round1.jsonl").read_text(encoding="utf-8").splitlines()]
    archived = next(r["request"] for r in rows if r["id"] == record["id"])
    control = json.loads(original)
    control.update({k: v for k, v in archived.items() if k in control})
    (out / "request_0001.json").write_text(json.dumps(control, ensure_ascii=False), encoding="utf-8")
control = json.loads((out / "request_0001.json").read_text(encoding="utf-8"))
assert hashlib.sha256(control["prompt"].encode()).hexdigest() == record["prompt_sha256"]
assert {k: v for k, v in control.items() if k != "prompt"} == {
    k: v for k, v in json.loads(original).items() if k != "prompt"
}
PY

git rev-parse HEAD > "$HY3_RUN/gpu_acceptance/revision.txt"
git status --short >> "$HY3_RUN/gpu_acceptance/revision.txt"
nvidia-smi > "$HY3_RUN/gpu_acceptance/gpu.txt"
python -m pip freeze > "$HY3_RUN/gpu_acceptance/packages.txt"
sha256sum /new_data/HunyuanImage-3.0-Instruct/{config,tokenizer,generation_config}.json \
  > "$HY3_RUN/gpu_acceptance/model-hashes.txt"
cp experiments/hunyuan_pd/configs/npd.yaml "$HY3_RUN/gpu_acceptance/npd-gpu.yaml"
```

Also record the model weight revision/snapshot. The original tokenizer SHA-256 is
`c2439418b9be76a54d8b5ff1ed89b37e36ec1735730f7da99b5aabb83a73db64`.
Do not replace the checkpoint with an FP8/Distil model for this acceptance.

`request_0007.json` retains this original body, including the prompt newline:

```json
{
  "prompt": "成年东方女性穿带完整内衬的幻彩未来感连衣裙，长发在浅紫、冰蓝和淡粉间渐变。她自然站立，侧颜安静，百褶裙与结构拼接反射珠光，简洁背景，柔光突出发丝和服装材质。\n采用1:1方形画幅，主体完整置于画面内，四周留出适当空间。",
  "size": "1024x1024",
  "n": 1,
  "num_inference_steps": 50,
  "guidance_scale": 0.0,
  "seed": 42,
  "bot_task": "think",
  "use_system_prompt": "en_unified",
  "model": "/new_data/HunyuanImage-3.0-Instruct/",
  "response_format": "b64_json"
}
```

## Run one GPU functional acceptance

Apply the small [temporary probe](gpu_reuse_probe.patch) before starting the
server. It adds assertions and JSON evidence only for this single-request,
50-step acceptance; it changes no token/KV values or inference decisions.
The probe synchronizes token reads and logs each layer, so this is not a
performance measurement. Warmup requests are excluded.

```bash
git apply --check tests/diffusion/models/hunyuan_image3/gpu_reuse_probe.patch
git apply tests/diffusion/models/hunyuan_image3/gpu_reuse_probe.patch
git diff -- vllm_omni/diffusion/models/hunyuan_image3 \
  > "$HY3_RUN/gpu_acceptance/runtime-probe.diff"
```

Start with the **original server command**, after preserving any old
`npd-server.log`. This is a foreground command; use a second terminal for replay.
Ensure the environment imports this checkout (for example, an editable install).

```bash
vllm serve /new_data/HunyuanImage-3.0-Instruct/ --omni \
  --port 9000 --stage-configs-path experiments/hunyuan_pd/configs/npd.yaml \
  --log-stats > /home/w00934428/hy3_run01/npd-server.log 2>&1
```

After the server is ready, replay the two original requests **serially**, with
no other traffic. Preserve an individual log interval and response for each.

```bash
export HY3_RUN=/home/w00934428/hy3_run01
curl --fail --silent --show-error http://localhost:9000/health
for case_id in 0007 0001; do
  start_byte=$(wc -c < "$HY3_RUN/npd-server.log")
  curl --max-time 1800 --silent --show-error \
    -D "$HY3_RUN/gpu_acceptance/$case_id.headers" \
    -o "$HY3_RUN/gpu_acceptance/$case_id.response.json" \
    -w '%{http_code}\n' \
    http://localhost:9000/v1/images/generations \
    -H 'Content-Type: application/json' \
    --data-binary "@$HY3_RUN/gpu_acceptance/request_$case_id.json" \
    > "$HY3_RUN/gpu_acceptance/$case_id.status"
  tail -c +"$((start_byte + 1))" "$HY3_RUN/npd-server.log" \
    > "$HY3_RUN/gpu_acceptance/$case_id.server.log"
done
```

The original batch-driver command is retained below for reference. It requires
the original experiment script and schedule, which are absent from this main
checkout. It is unnecessary for the two serial acceptance requests above;
recreating C=12 concurrency is not the GPU acceptance gate.

```bash
python experiments/hunyuan_pd/run_experiment.py \
  --deployment npd \
  --schedule-dir /home/w00934428/hy3_run01/schedule_linux \
  --output-root /home/w00934428/hy3_run01 \
  --server-log /home/w00934428/hy3_run01/npd-server.log \
  --timeout 1800
```

## Prove that AR KV was reused

The probe emits `HY3_GPU_ACCEPT` JSON at four actual execution points:

| Event | Required evidence |
| --- | --- |
| `layout` | Original AR `P + G` equals DiT's prefix, element by element; `R = len(P + G) - 1 > 0`; all layers' K/V cover `R`; classify actual `G` against `encode(decode(G))`. |
| `truncate` | After `_maybe_handle_ar_kv_reuse`, first query token is `</think>` (`128024`); query length is `L - R`. |
| `consume` | Each layer's first `_cache_prompt_kv` consumes nonzero `_injected_ar_kv`; actual new Q length plus consumed R equals L; concatenated K/V both have length L. |
| `denoise_done` | All 50 denoising iterations completed; HTTP/image checks below additionally cover output decoding. |

These are assertions on runtime values, not expected lengths copied from the
CPU fixture. `Handling AR KV reuse...` alone is only an intent log. The
`consume` records establish that imported KV reached every layer; `Q < L`
establishes that the first step did not recompute the full prefix.

Run the following checks in the same second terminal. They require records for
all four DiT ranks, every layer, both requests, and an actual 1024 x 1024 image.

```bash
python - <<'PY'
import base64
import io
import json
import os
from pathlib import Path
from PIL import Image

out = Path(os.environ["HY3_RUN"]) / "gpu_acceptance"
classes = []
for case_id in ("0007", "0001"):
    assert (out / f"{case_id}.status").read_text(encoding="utf-8").strip() == "200", case_id
    response = json.loads((out / f"{case_id}.response.json").read_text(encoding="utf-8"))
    assert len(response["data"]) == 1
    encoded = response["data"][0]["b64_json"].split(",")[-1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    image.load()
    assert image.size == (1024, 1024)
    image.save(out / f"{case_id}.png")
    events = []
    for line in (out / f"{case_id}.server.log").read_text(encoding="utf-8").splitlines():
        if "HY3_GPU_ACCEPT " in line:
            events.append(json.loads(line.split("HY3_GPU_ACCEPT ", 1)[1]))
    assert events and len({e["request_id"] for e in events}) == 1, "Missing/mixed request evidence"
    assert {e["rank"] for e in events} == {0, 1, 2, 3}, "Missing DiT rank evidence"
    layouts = [e for e in events if e["event"] == "layout"]
    assert len({e["roundtrip_equal"] for e in layouts}) == 1
    for rank in range(4):
        group = [e for e in events if e["rank"] == rank]
        layout, = [e for e in group if e["event"] == "layout"]
        trunc, = [e for e in group if e["event"] == "truncate"]
        done, = [e for e in group if e["event"] == "denoise_done"]
        r, length = layout["reuse_len"], layout["full_len"]
        assert layout["prefix_equal"] and r == layout["prompt_len"] + layout["generated_len"] - 1 > 0
        assert layout["kv_min_len"] >= r and layout["end_think_id"] == 128024
        assert trunc["reuse_len"] == r and trunc["first_uncached_id"] == 128024
        assert trunc["query_len"] == length - r < length and done["steps"] == 50
        consumed = [e for e in group if e["event"] == "consume"]
        assert len(consumed) == layout["layers"]
        assert {e["layer"] for e in consumed} == set(range(layout["layers"]))
        for e in consumed:
            assert e["reuse_len"] == r and e["query_len"] == trunc["query_len"]
            assert e["reuse_len"] + e["query_len"] == e["full_len"] == length
            assert e["key_len"] == e["value_len"] == length
    stable = layouts[0]["roundtrip_equal"]
    classes.append(stable)
    print(case_id, "reuse PASS;", "stable control" if stable else "non-roundtrip trigger")
assert set(classes) == {False, True}, "Reuse passed, but trigger/control coverage is incomplete"
print("CPU regressions pass, and the target AR KV reuse path completes real inference in a main-based GPU environment.")
PY
```

Neither the original failure nor its control is guaranteed to produce the same
AR IDs on GPU. If both runs classify as stable, record **reuse passed, trigger
coverage pending**; if both change, the stable control is pending. Keep the
original prompt, seed, and request settings. Further candidates must come from
the archived original requests and be classified from their actual GPU outputs.
Do not feed saved tokens or dummy KV in place of real AR inference. The original
`R=1777`, `Q=4105`, `L=5882` is a reference only when the full original token
sequence is reproduced.

After saving the logs, responses, images, revisions, configuration and probe
diff, stop the server and remove only the temporary probe:

```bash
git apply --reverse --check tests/diffusion/models/hunyuan_image3/gpu_reuse_probe.patch
git apply --reverse tests/diffusion/models/hunyuan_image3/gpu_reuse_probe.patch
```

Only after both token classes pass all checks, report: **CPU 回归通过，主干 GPU
上目标 AR KV 复用路径完成真实推理。** This does not claim other execution modes,
image quality improvements, or a speedup.
