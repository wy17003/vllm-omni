# HunyuanImage3 A1：P 同步、D 异步

范围：NPU、1P1D、每阶段 TP=4、temperature=0、n=1、PP/CP=1，无推测解码或 resumable 输入。
P 仍在第一次采样结果返回时结束阶段，因此继续拒绝异步 P。D 复用上游 AsyncScheduler，
导入的 y1 是已确认生成历史，只有新调度的采样占用输出 placeholder。

## 配置与请求

沿用原服务启动命令，每次切换下面 YAML 后重启，保存启动日志及全部 worker 请求日志。
路径均位于 `vllm_omni/deploy/`；同步到服务器时包含全部 base_config 依赖。

| 顺序 | YAML | 请求 | 预期 |
| --- | --- | --- | --- |
| 1 | `hunyuan_image_3_moe_pd_ar_async_decode_max_tokens_1.yaml` | 原猫请求 | AR 输出 `[791]`；D 接收 KV 后结束，无 AR forward |
| 2 | `hunyuan_image_3_moe_pd_ar_async_decode_max_tokens_2.yaml` | 原猫请求 | AR 输出 `[791,1217]`；D 一次有效 AR forward |
| 3 | `hunyuan_image_3_moe_pd_ar_async_decode.yaml` | 原猫请求 | 完整 900 token 与同步基线一致 |
| 4 | `hunyuan_image_3_moe_pd_ar_async_decode.yaml` | 完整 prompt 长度 1280 的请求 | 完整 AR 输出与同请求的同步基线一致 |

原猫请求和长度请求生成方法见[最小回归实验](hunyuan_image3_pd_minimal_regression.md)。
每例先运行一次，任一失败先定位该例；通过后可在同一服务实例再重复一次。
前两项仅验收 AR 接续与终止，不评价截断 CoT 后的图像质量。

三份 A1 配置均显式设置 P `async_scheduling=false`、D `async_scheduling=true`。
底层继承同步实验配置以保留 Mooncake 环境变量、NPU 布局和现有只读日志，
`VLLM_OMNI_HY3_KV_CAUSAL_MODE=off`，不启用旧尾 KV 因果探针。
基础 PD YAML 中省略 D 的 async 字段并不能代替本实验的显式配置。

## 验收

首先确认服务实际解析的 stage 0 使用 OmniARScheduler、stage 1 使用 OmniARAsyncScheduler，
且没有启动脚本/CLI 覆盖 D 为同步。输出相同本身不能证明异步已启用。

按 request ID、整体输出序号和 TP rank 比较现有日志：

- 正常接续的 D 首次输入是 P 的 y1；原猫请求为 token 791、位置 1236、历史计数 1。
- D 第一次新采样为整体第 2 个输出，max_tokens=2 的第二个 token 为 1217。
- max_tokens=1 必须有最终 `[791]` 输出和正常结束，不能只用“没有 D 日志”作为通过证据。
- 完整猫请求的输出 SHA256 为 `054e0d83c7f7ba11fc17f66dd5ee755c57dccae019b1ab0f21ee69f764518060`。
- AR 导出有效 KV 长度与同步基线一致；普通输出 N 个 token 时为 L+N-1。
  已确认 KV 长度排除尚在途的输出 placeholders，不能直接采用调度器乐观计算长度。
- 无重复首 token、重复终止输出、占位计数下溢或请求挂起。异步下物理执行可有在途步骤，
  它们的迟到结果不能进入已经结束的请求输出，也不能计入该请求导出的有效 KV。

同步对照仍使用 `hunyuan_image_3_moe_pd_ar_max_tokens_1.yaml`、
`hunyuan_image_3_moe_pd_ar_max_tokens_2.yaml`、`hunyuan_image_3_moe_pd_ar_boundary.yaml`，无需改动。
本轮只评估功能，现有诊断同步开销及重复阶段统计使日志不适合性能结论。

## 本地与服务器检查

```bash
pytest -q tests/engine/test_pd_continuation.py tests/test_omni_request.py tests/entrypoints/test_omni_new_request_data.py
```

CPU 回归覆盖异步 D 接入、异步 P 拒绝、其他限制保留、完整 KV 接收、导入历史、
占位计数和在途 KV 长度、batch 换位后的采样历史恢复，以及无 forward 终止和迟到结果。
本地使用源码隔离环境执行相关真实方法；它不覆盖 NPU stream 执行和 Mooncake 传输时序，
后者由上面的服务器实验验证。
