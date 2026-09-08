# HunyuanImage3：prompt 尾部 KV 因果实验（NPU，1P1D）

目的：验证仅恢复 consumer 首次重算覆盖的 prompt 尾部 KV，是否足以消除
PD 与非 PD 的后续 AR logits/token 差异。本代码是实验插桩，不是正式的 PD 接续修复。

## 实验动作

1. Producer 完整 prefill 后，在每个 TP rank 上遍历所有 KV 层，对有效 prompt
   的前缀 `[0, L-1)` 和尾部 `L-1` 分别计算完整原始字节 SHA256。
   只将尾部张量保存在 CPU；首次采样后，把尾部、校验信息及实际首 token 原子写入快照文件。
2. Consumer 首次 forward 必须满足 `num_computed_before=L-1, num_scheduled=1`。
   forward 结束后读取同一请求、同一 TP rank 的 producer 快照；逐层校验所有前缀元素。
   这里校验的是**重算后没有被本次 forward 写入的前缀**，不是传输入口的前置探针。
3. `check` 模式记录差异、不写 KV。`restore` 模式在所有层验证通过后，覆盖
   consumer 位置 `L-1` 的全部层 K/V，再读回并逐字节校验。
   当前样例 `L=1236`，所以恢复位置自动为 `1235`，没有硬编码位置。
4. 两种模式均保留 consumer 首次 forward 的 hidden states、logits 和采样结果，
   并核对首 token 与 producer 是否一致。恢复影响从**第 2 个输出 token 的计算**开始体现。

文件按内部 request ID 和 TP rank 命名，并校验 prompt SHA、长度、TP 大小和张量布局。
P/D 必须能访问同一个目录（当前同机部署适用；不同容器需挂载同一目录）。
成功核对首 token 后，consumer 删除自己的快照；失败时保留用于诊断。
请求内容相同即可，三组实验间不需要固定内部 request ID。

## 运行顺序

保持现有服务器启动命令、模型和请求内容，分别把部署 YAML 换成下表配置；每组重新启动服务，
串行执行相同请求，保存所有 worker 的完整日志。日志名只是建议。

| 顺序 | YAML（位于 `vllm_omni/deploy/`） | 日志 | 用途 |
| --- | --- | --- | --- |
| A | `hunyuan_image_3_moe_kv_check.yaml` | `non_pd_check.log` | 非 PD 全量前缀/尾部指纹基线 |
| B | `hunyuan_image_3_moe_pd_kv_check.yaml` | `pd_check.log` | PD 只校验对照 |
| C | `hunyuan_image_3_moe_pd_kv_restore.yaml` | `pd_restore.log` | PD 恢复尾部 KV |

这三个薄配置继承原配置，保留 NPU 的每阶段 4 卡布局；AR 统一设为同步调度。
**先确认 A/B 仍能复现输出差异**，再用 C 归因。若 A/B 已一致，应先分析同步调度或插桩
引入的同步是否改变了现象，此时不能说恢复 KV 解决了问题。

实验要求：batch=1、PP/CP=1、仅 TP、无 EP/推测解码，P 和非 PD 的 prompt 在一次 forward
中完整处理、D 首次仅重算一个尾 token。当前 1236-token 请求符合 8192-token 调度预算。
不支持上述条件的运行会明确失败。不要用这些插桩结果衡量性能。

PD 两个 AR stage 的 `VLLM_OMNI_HY3_KV_CAUSAL_DIR` 必须完全相同且可写。
默认 B 为 `/tmp/vllm-omni-hy3-kv-check`，C 为 `/tmp/vllm-omni-hy3-kv-restore`。
失败后若重用相同内部 request ID，请换一个新的实验目录，并同步修改 P/D 两处配置。
部署 YAML 的 `env` 在 `base_config` 继承时整块覆盖，新增配置已保留原来的 debug 和 Mooncake 环境变量。

现有请求的客户端超时建议从 180 秒提高到 600 秒，以容纳第一次全量校验的额外开销。
至少运行一组 A/B/C；若结果支持假设，可再重复同样请求确认。

## 日志验收

新增日志前缀为 `[HY3_KV_CAUSAL]`，后面是 JSON，每条都有 `request_id/role/mode/tp_rank`。
按逻辑 rank 0–3、layer 和 key/value 对齐；不要按日志时间或物理 block ID 对齐。

| event | 每个 rank 应检查的信息 |
| --- | --- |
| `post_forward` | `metadata.prompt_sha256/prompt_len/layer_count`；全部层 `entries` 中的 `prefix_sha256/tail_sha256` |
| `producer_ready` | 快照已完整落盘，`first_token` 为 producer 实际采样值 |
| `prefix_verified` | consumer 所有层完整前缀均一致；`changed` 列出恢复前尾部不同的层/KV 类型 |
| `tail_restored` | 仅 C 出现；`readback_equal=true`，`tail_position=1235`，全部尾部 SHA 与 producer 对齐 |
| `first_token_check` | B/C 均须 `equal=true`；当前请求预期首 token 为 791 |

当前模型预期每个 rank `layer_count=32`、`kv_tensor_count=64`，4 个 rank 合计 256 个 K/V 张量。
缺少任何 rank 的成功日志，都不能宣布恢复成功。`invalid experiment`、前缀不等或首 token 不等，
均表示本次因果实验不满足条件；异常会使实验运行失败，不会悄悄退回普通推理。

跨 A/B/C 比较 producer 与非 PD 的所有层前缀及尾部 SHA，而不仅是此前的固定位置抽样。
这一步需从日志对齐检查；程序内部自动检查的是同一次 PD 请求的 P/D 前缀与恢复结果。

继续使用现有 `[HY3_AR_LOGITS]`，重点比较第 2 个输出和第 94 个输出的
token、top-2、分数及 history SHA；此处按输出序号从 1 计数，日志 step 按实际字段定义对齐。
默认保留前 101 步日志。最终用 `[HY3_EQ] ar` 的 `output_token_count`、
`output_token_sha256` 和完整 `output_token_ids` 比较整个 AR 序列，不以最终图像代替 AR 验收。

## 结论边界

- A/B 仍不同，所有前提校验通过，C 后续 logits 和完整 AR tokens 与 A 一致：
  支持“该请求的尾部 KV 差异足以解释后续 AR 差异”。接下来再设计避免尾 token 重算的正式接续修复。
- C tokens 一致但 logits 仍不同：只证明本次离散输出恢复一致，数值等价仍未完全恢复。
- C 仍不同：尾部 KV 不是充分解释；在完整前缀验证结果基础上继续检查其他状态或算子路径。
- A/B 不复现、任一前缀校验失败、首 token 不同、rank 日志缺失：不满足归因条件，应先定位该项。

关闭实验时恢复原部署 YAML；`VLLM_OMNI_HY3_KV_CAUSAL_MODE` 默认 `off`，不交换文件、不改 KV。
本地 CPU 单测覆盖分页映射、BF16、rank 隔离、完整前缀校验、只写尾部及异常拒绝；
它们不能代替服务器上的 NPU/Mooncake 端到端验证。
