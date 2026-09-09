# HunyuanImage3 A2：固定 seed 随机采样接续验收

## 修改与参数来源

P 在同步 worker 的 `_bookkeeping_sync` 之后，只为被接受的首 token 导出请求 generator
的完整状态。无效的分块 prefill 采样不导出。各 TP rank 的状态通过 TP CPU group 汇总，
以版本化字节载荷随 P 输出的 `kv_transfer_params.pd_rng_state` 返回。
Orchestrator 将载荷移入 `PDContinuation.rng_state`，从发给 Mooncake 的参数中移除该键。
`OmniNewRequestData.pd_rng_state` 将载荷送到所有 D worker。

D 创建请求 generator 时，先检查 seed、设备类型和 TP 数量，再恢复自身逻辑 rank 的状态。
恢复发生在请求进入 input batch 前；以后 decode 使用同一 generator，不重复恢复。
CPU/NPU/CUDA 的状态都按原始字节传递，不通过硬编码 offset 或额外抽样推进状态。
缺失状态或参数不一致会报错，不能静默退回从 seed 重新开始。

P 继续同步，D 保留异步。首 token 直接满足 max_tokens/EOS/stop 时，D 仍由 scheduler
完成请求，无需 worker forward 或 RNG 恢复。已有 worker 请求在抢占后恢复时保留其当前
generator；若尝试丢失 worker 状态后用首 token 的 RNG 重建已有更多输出的请求，会明确拒绝。

逻辑 AR 的 `default_sampling_params` 只配置在 D；P 的对外默认参数和每次提交参数都从 D
克隆，运行时再加入 P 的单次采样和 KV 传输标记。没有引入顶层通用 sampling YAML 字段，
避免与 DiT 参数混淆。优先级为：请求对 D 的有效覆盖 > D 的 YAML 默认值；P 的独立采样值
不参与该接续模式。图像 API 的请求 seed 同时应用到 P/D，再由 D 派生 P。
P 导出的 RNG 必须匹配该有效 seed，D 使用导出状态继续采样，不能用 seed 覆盖状态。

正式接续支持 `temperature>0` 且显式 seed；仍要求 n=1、无 logprobs/grammar、PP/CP=1、
无推测解码、无 resumable input。旧的 `pd_rng_repro_without_state: true` 仅用于重现缺陷。

## 本轮最小操作

先在服务器执行：

```bash
pytest -q tests/engine/test_pd_continuation.py tests/engine/test_pd_orchestrator.py \
  tests/worker/test_pd_rng.py tests/utils/test_hunyuan_rng_debug.py -m "not gpu"
```

分别部署以下 YAML，沿用原启动指令、NPU 布局和完全相同的请求：

| 模式 | YAML（vllm_omni/deploy 下） | 建议日志 |
| --- | --- | --- |
| 非 PD | hunyuan_image_3_moe_rng_check.yaml | non_pd_rng_check.log |
| PD 修复 | hunyuan_image_3_moe_pd_rng_check.yaml | pd_rng_check.log |

两者仍为 temperature=1、seed=42、top_p=1、top_k=-1、max_tokens=32，NPU 每阶段4卡。
每种模式连续请求两次。请求体沿用 `hunyuan_image3_rng_reproduction.md` 中的猫请求。
验收配置继承复现配置并明确覆盖 `pd_rng_repro_without_state: false`，注意不要误用旧 PD
repro YAML；旧 YAML 仍然故意跳过状态传递。

按 request_id、逻辑 TP rank 对齐，验收条件如下：

1. prompt IDs、有效 AR seed、前两步完整 logits 指纹一致。
2. 新日志 `[PD_RNG_STATE] event=export` 与同 rank 的 `event=restore` 的 state_sha256 一致。
   `[HY3_AR_RNG]` 中 D 第2次整体输出的 rng_before，应等于 P 的 rng_after 以及非 PD 第2步
   rng_before。本例预期 offset 为12，采样后为24；判定以实际完整状态指纹为准。
3. 两者完整32个 AR output_token_ids 一致，各自两次也一致。PD 首3个相同、第4个不同
   的旧现象应消失。

完成后，将请求 seed 同时改为43，各模式再请求一次，确认日志 initial_seed=43，验证请求覆盖
默认42以及跨模式一致性。不要求不同 seed 的文本必然不同。

如需补测随机终止边界，将两个验收 YAML 对应 AR stage 的 `max_tokens` 同时设为1，再设为2。
PD 只改 D，非 PD 改 stage 0：上限1应仅输出 P 首 token，D 无采样；上限2应恰好输出两个
token，首 token 只出现一次，并与非 PD 完全一致。异步在途计算不计入最终输出数。

本轮只验收 AR。32-token 截断 CoT 后的 DiT 输出不作为判据；RNG debug 会同步设备，不能用于
性能测量。本地 CPU 隔离测试覆盖状态/序列/序列化/边界逻辑，实际 NPU sampler、TP collective
和 Mooncake 传递链仍以这轮服务器实验为准。
