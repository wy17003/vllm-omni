# HunyuanImage3 PD 首 token 接续：最小回归实验

适用范围：NPU、1P1D、每阶段 TP=4、temperature=0、同步 AR 调度。
同请求连续重复已通过，本轮只做三组实验，共 10 次请求。

## 配置与运行顺序

保持现有启动命令和模型路径，每次切换下表 YAML 后重启服务。
所有 YAML 均位于 `vllm_omni/deploy/`，需连同其 `base_config` 依赖一起同步到服务器。

| 实验 | 非 PD YAML | PD YAML | 请求及次数 |
| --- | --- | --- | --- |
| 首 token 即结束 | `hunyuan_image_3_moe_ar_max_tokens_1.yaml` | `hunyuan_image_3_moe_pd_ar_max_tokens_1.yaml` | 原猫请求，两种模式各 1 次 |
| 接续一步后结束 | `hunyuan_image_3_moe_ar_max_tokens_2.yaml` | `hunyuan_image_3_moe_pd_ar_max_tokens_2.yaml` | 原猫请求，两种模式各 1 次 |
| KV block 边界 | `hunyuan_image_3_moe_ar_boundary.yaml` | `hunyuan_image_3_moe_pd_ar_boundary.yaml` | 完整 prompt 长度 1279/1280/1281，两种模式每例各 1 次 |

薄配置采用与 `hunyuan_image_3_moe_kv_check.yaml` 相同的继承方式，但直接继承正式配置，
显式设置 `VLLM_OMNI_HY3_KV_CAUSAL_MODE=off`。旧 check/restore 探针要求 D 重算 prompt 尾 token，
与当前接续路径不兼容；本轮不使用旧因果实验配置或快照目录，也不修改日志插桩。
若启动脚本另行注入该开关，也将其设为 `off`。

继承后 NPU 布局为：非 PD AR 8–11、DiT 12–15；PD P 4–7、D 8–11、DiT 12–15。
两种模式串行启动，避免设备及端口冲突。保留当前 eager、单请求调度和既有只读诊断。
`env` 在 YAML 继承时整块覆盖，因此 PD 配置完整保留了 Mooncake 的 IP 和 bootstrap 端口。

`max_tokens` 修改的是非 PD 的 stage 0、PD 的 stage 1；P 从 D 获取逻辑参数，并在一次采样后移交。
`min_tokens=0`，PD 的 `pd_resume_from_prefill=true` 通过继承保留。
图像接口不直接提供 AR `max_tokens` 参数，不要在请求 JSON 中增加该字段代替配置切换。

## 请求

前两组使用已通过实验的原请求：

```bash
curl --fail-with-body --max-time 300 http://localhost:9000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "generate a sleeping cat",
    "model": "/new_data/HunyuanImage-3.0-Instruct/",
    "use_system_prompt": "en_unified",
    "bot_task": "think",
    "num_inference_steps": 50,
    "n": 1,
    "seed": 42
  }' -o response.json
```

每次将响应和服务日志另存为对应实验、模式的文件，避免覆盖。
长度边界用例在服务器仓库根目录生成一次，PD 和非 PD 复用相同文件：

```bash
python - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import build_prompt_tokens

model = "/new_data/HunyuanImage-3.0-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
out = Path("pd_validation_requests")
out.mkdir(exist_ok=True)
targets = {1279, 1280, 1281}
found = set()
for count in range(512):
    prompt = "generate a sleeping cat" + " soft" * count
    result = build_prompt_tokens(
        prompt, tokenizer, task="t2i", bot_task="think", sys_type="en_unified"
    )
    length = len(result.token_ids)
    if length not in targets or length in found:
        continue
    body = dict(
        prompt=prompt, model=model, use_system_prompt="en_unified",
        bot_task="think", num_inference_steps=50, n=1, seed=42,
    )
    path = out / f"prompt_{length}.json"
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    found.add(length)
    print(length, path)
    if found == targets:
        break
if found != targets:
    raise RuntimeError(f"Missing target lengths: {sorted(targets - found)}")
PY
```

分别启动非 PD 和 PD 的 `*_ar_boundary.yaml` 后执行下面命令；设置不同 `mode` 以区分响应：

```bash
mode=non_pd  # PD 服务运行时改为 pd
for length in 1279 1280 1281; do
  curl --fail-with-body --max-time 300 http://localhost:9000/v1/images/generations \
    -H 'Content-Type: application/json' \
    --data-binary "@pd_validation_requests/prompt_${length}.json" \
    -o "response_${mode}_${length}.json" || break
done
```

以服务日志的 `prompt_token_count` 确认实际长度。预期 KV block_size=128；若实际不同，
1279/1280/1281 不再代表上述 block 边界，需要围绕实际 block_size 的整数倍重新选取长度。

## 最小验收

| 实验 | 预期 AR 结果 | 额外检查 |
| --- | --- | --- |
| max_tokens=1 | 原猫请求两种模式均为 `[791]` | D 接收完整 prompt KV 后结束，无 AR forward/采样；有最终输出、无挂起 |
| max_tokens=2 | 原猫请求两种模式均为 `[791,1217]` | D 首次输入 791、位置 1236、历史计数 1，采样整体第 2 个输出后结束 |
| 长度边界 | 每个请求的 PD/非 PD 完整输出 token IDs 相同 | D 首次 `num_computed_before=L`、位置 L、历史计数 1，输入等于 P 的 y1 |

按请求和 TP rank 0–3 对齐 `HY3_AR_INPUT`、`HY3_AR_LOGITS`，用 `HY3_EQ ar` 的完整 token IDs
和 hash 比较最终结果。不同长度请求之间不要求输出相同；本轮没有全量 KV 因果校验日志。
不能仅凭 D 没有日志判断 max_tokens=1 通过，还必须确认 AR 输出及正常结束。

max_tokens=1/2 会截断 CoT/ratio，只验收 AR 接续及终止，不评价图像质量。
若后续 DiT 出错，单独记录 AR 是否完成和故障阶段；HTTP 200 本身也不能代替 AR token 校验。
当前统计存在重复阶段事件，不以 e2e_total_tokens 验收输出数量，不用本轮诊断运行比较性能。
