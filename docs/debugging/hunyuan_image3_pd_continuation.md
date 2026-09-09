# HunyuanImage3 PD 首 token 接续

`hunyuan_image_3_moe_pd.yaml` 的 decode stage 默认启用
`default_sampling_params.extra_args.pd_resume_from_prefill: true`，两个 AR stage 使用同步调度。
当前范围为单输出、`temperature=0`、PP/CP=1，无推测解码；暂不支持 logprobs 和结构化输出状态接续。

## 状态约定

Producer 使用 decode stage 的逻辑采样参数进行一次采样，得到 `y1`。
保留原始 min/max token、EOS 和 penalty 参数；调度器通过独立的阶段执行边界结束 producer，
以 `length` 原因满足 Mooncake 的 KV 保留协议。因此日志中 P 的逻辑 `max_tokens` 可以大于 1，
但实际仍只采样一次。文本停止条件由 consumer 输出处理器在提交请求前检查。

Consumer 请求携带可序列化的 `PDContinuation`，原始 prompt 不追加 `y1`。
创建请求时把 `y1` 纳入生成历史和完整逻辑序列，但不声称 KV 已就绪：

| 状态 | 当前样例 |
| --- | --- |
| prompt 长度 | 1236 |
| 初始生成历史 | `[791]` |
| 完整逻辑序列长度 | 1237 |
| KV 接收确认后的已计算长度 | 1236 |
| 首次 consumer forward 输入/位置 | `791` / `1236` |
| 首次 consumer 新采样输出 | 整体第 2 个输出 |

上游 Mooncake 加载完整 prompt KV；上游 scheduler 的“已计算整个序列则回退一个 token”条件
此时不成立。接收完成后的校验会拒绝不完整 KV，避免静默退回尾 token 重算。

Scheduler、worker、采样器从同一生成历史初始化。Scheduler 仅在首次发出的输出增量中补上
producer token；detokenizer、累计文本、使用量及 AR 桥接沿用现有累计流程，不二次拼接。
逻辑 `max_tokens=N` 保持不变，consumer 已有一个输出，因此最多再生成 `N-1` 个。

若 `y1` 已满足 `max_tokens=1`、EOS、停止 token 或字符串，consumer 等待 prompt KV 接收完成，
通过现有释放/下游提取流程结束请求，不执行额外 forward。停止字符串由正常 detokenizer
最终应用，保留同一 token 中停止字符串之前的文字。

## 服务器验证

原请求重复验证通过后，使用[最小回归实验](hunyuan_image3_pd_minimal_regression.md)中的六份薄配置，
完成 max_tokens=1/2 和 prompt 长度 1279/1280/1281 的十次请求验证。

使用原启动命令并指定 `vllm_omni/deploy/hunyuan_image_3_moe_pd.yaml`，继续发送同一个请求。
旧 KV 因果实验配置及探针没有调整；不要使用 `*_kv_check.yaml`、`*_kv_restore.yaml` 启动此接续路径。
若启动环境仍显式设置 `VLLM_OMNI_HY3_KV_CAUSAL_MODE`，将其设为 `off`。
现有 `HY3_AR_INPUT`、`HY3_AR_LOGITS`、`HY3_EQ` 日志保留原实现。

预期 consumer 首次输入位置为 1236，token 为 791，首次采样记录的 `generated_count_before=1`、
`output_ordinal=2`。最终完整 AR token 应与非 PD 的 900-token 序列一致。

在已安装本分支所需 vLLM 的服务器环境执行：

```bash
pytest -q tests/engine/test_pd_continuation.py tests/test_omni_request.py tests/entrypoints/test_omni_new_request_data.py
```

再覆盖 `max_tokens=1/2`、终止 token/字符串、连续请求及取消后的资源释放。
本地验证采用隔离加载相关源码的 CPU 环境，使用实际上游 SamplingParams、Request、
停止检查、KV 接收方法和 detokenizer 逻辑；未执行 NPU/Mooncake 服务集成测试。
