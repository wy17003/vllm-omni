# HunyuanImage3 A2 第一步：固定 seed 随机采样差异复现

> 本文保留第一步的缺陷复现操作。正式修复后的验收请使用
> [固定 seed 随机采样接续验收](hunyuan_image3_rng_validation.md)。当前正式路径已允许固定 seed
> 随机采样；下文“正式配置拒绝”描述的是修复前行为，旧 repro YAML 仍故意绕过状态接续。

本轮只复现并记录 RNG 接续缺口，不传递、恢复或推进 D 的 RNG 状态。
正式配置继续拒绝随机 PD 接续；仅实验参数 `pd_rng_repro_without_state: true`
临时允许带显式 seed 的随机采样，n=1、无 logprobs/grammar 等限制保持不变。

## 最小实验：四次请求

分别使用以下配置串行启动服务，保持原启动命令及模型路径。每种模式在同一服务实例
连续请求两次，保存启动日志、全部 TP worker 日志和两份响应。不要求跨模式 request ID 相同。

| 模式 | YAML（位于 vllm_omni/deploy） | 服务日志建议名称 |
| --- | --- | --- |
| 非 PD | `hunyuan_image_3_moe_rng_repro.yaml` | `non_pd_rng_repro.log` |
| PD | `hunyuan_image_3_moe_pd_rng_repro.yaml` | `pd_rng_repro.log` |

配置继承 NPU 每阶段 TP=4；非 PD AR 与 PD D 都显式使用异步调度，PD P 保持同步。
保持已通过的首 token 接续路径，关闭旧尾 KV check/restore 探针。
PD 的有效采样参数从 D 复制给 P，包括 seed；本轮未重构通用配置机制。

两种模式均设置 temperature=1、top_p=1、top_k=-1、seed=42、max_tokens=32、min_tokens=0，
不施加 repetition/presence/frequency 惩罚。32 是输出上限；遇到正常停止条件仍会提前结束。
实际生效的 AR seed 应从 RNG 日志的 initial_seed 核对，API 入口的 stage_sampling 日志在 P 参数派生之前。

```bash
mode=non_pd  # PD 服务启动后改为 pd
for run in 1 2; do
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
    }' -o "response_${mode}_rng_${run}.json" || break
done
```

图像接口的 JSON 不控制 AR temperature/max_tokens，请使用上述 YAML。
32 个 token 会截断 CoT，本轮只评价 AR；若 DiT 失败，保留 AR 和错误日志单独分析，
不要把图像错误当成 RNG 接续差异。若请求失败而循环中断，先检查日志再决定是否重试。

## 新日志及观察位置

`VLLM_OMNI_HY3_AR_RNG_DEBUG=1` 独立启用 `[HY3_AR_RNG]`，只记录整体前两个输出的采样。
P 对应 output_ordinal=1，D 从 2 开始；非 PD 对应 1、2。按 request_id、role、tp_rank 和序号对齐。
已有 `[HY3_AR_LOGITS]` 记录前 32 个输出，`[HY3_EQ] ar` 给出完整输出列表和 hash。

| RNG 日志字段 | 含义 |
| --- | --- |
| rng_before / rng_after | 请求级 generator 在实际 sampler 调用前后的状态摘要；包含 initial_seed、state_sha256、state_numel，后端支持时包含 offset |
| rng_at_random_sample | top-k/top-p 采样模块入口的 generator 状态 |
| model_sampler_input_logits | Hunyuan 模型约束处理后、通用 sampler 处理前的全词表指纹 |
| random_sampler_input_logits | 通用 sampler 的惩罚、temperature 等处理后，top-k/top-p 过滤前的全词表指纹；本实验无 top-k/top-p 截断 |
| random_sampler_called / sampler_class / random_sampler_class | 确认实际进入随机采样模块及后端实现 |
| sampling_controls | 实际采样 metadata 中的 temperature、top-k/top-p 和惩罚参数；None 可表示对应操作已禁用 |
| sampled_token_ids / history_sha256 | 此次采样结果和采样前的生成历史 |

日志调用只读取 generator，不调用 set_state/manual_seed，不额外消耗随机数。
首两步会同步设备以读取诊断，不能用本实验评估异步调度性能。
NPU 的 get_state/get_offset 支持情况以服务器日志为准；error、hook 未触发或同步失败
均表示相应观测不完整，不能直接据此得出 RNG 归因结论。

event=sample_before_bookkeeping 表示“sampler 返回后、worker bookkeeping 前”。
NPU runner 会对被丢弃的 partial-prefill 采样回退 offset，因此本实验要求 P 和非 PD
的 1236-token prompt 一次完整 prefill（预算 8192），且无抢占/失败重算；
须结合现有 HY3_AR_INPUT 确认 num_computed_before=0、num_scheduled=1236。
异步终止后可能有在途计算，以最终有效输出序列判定，而不是简单累计 sampler 日志条数。

## 判定顺序

1. 检查两个模式实际 prompt IDs 相同、AR initial_seed=42、采样控制一致；各自两次输出是否相同。
2. 检查 P 与非 PD 首次采样前的 logits、RNG hash 及首 token 一致，采样后的 RNG hash 也一致。
3. 检查 D 首次随机采样（整体第 2 个输出）的输入 logits 与非 PD 第 2 步是否一致。
4. 对同一逻辑 TP rank，预期 D.rng_before 等于 P.rng_before，
   却不同于 P.rng_after / 非 PD 第 2 步的 rng_before：这表明 D 重新从 seed 起点采样。
   状态比较使用 seed、字节 hash 和 offset，不要求物理 device 字符串一致。
5. 比较最终 AR output_token_ids，记录首次 token 分歧序号。

第 2 个 token 不一定马上分歧；相同随机状态不代表两步的 logits 相同，因此也不要求 D 的 y2 等于 P 的 y1。
若 RNG 确实错位但 32 个输出偶然相同，先保留这个证据，再将两种模式上限同时改为 128，
或同时用请求 seed=43 重试。不要假设任意非零 temperature 都必然产生可见 token 差异。
若第一步 token/logits 就不同，或各自重复不稳定，先定位该项，不能直接归因于交接 RNG。

本轮完成后提供两份服务日志；确认复现证据后再实施正式 RNG 状态传递和统一参数来源，
用相同请求、参数和设备布局验证修复。退出实验时切回原 A1/同步配置，移除
pd_rng_repro_without_state 并关闭 RNG_DEBUG。

## 本地检查范围

新增 CPU 单测验证：实验开关默认拒绝、固定 seed 必需、其他限制保留，
以及读取指纹/临时采样 hook 不改变输出、RNG 状态或 logits，hook 在异常时清理。
CPU 模拟采样器用于验证插桩透明性，不代替实际 NPU sampler 和 Mooncake 实验。

```bash
pytest -q tests/engine/test_pd_continuation.py tests/utils/test_hunyuan_rng_debug.py
```
