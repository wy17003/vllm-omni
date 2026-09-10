# HunyuanImage-3.0：现有配置端到端等价验证

本轮只验证 NPU、1P1D、每阶段 TP=4、单请求 T2I（think）、CFG 关闭的完整链路。
保持 P `async_scheduling: false`，D 和非 PD AR `async_scheduling: true`；
保持 `step_execution: false`、KV 异步预取开启、prefix caching 关闭。
不再执行尾 KV restore，不打开旧 AR logits/RNG 定位探针，不更改已有接续修复。

此前随机采样只验证了 32 个 AR token；本轮恢复 `max_tokens: 8192` 和正常停止条件，
验证完整 AR 输出及后续 50 步 DiT、VAE 和返回图像。8192 是上限，不能以截断结果验收。

## 最小实验：4 次部署，每次连续请求 2 次

每次切换配置都停止上一服务；同次部署的两次请求串行发送，不重启。
所有请求均使用 `generate a sleeping cat`，其余请求字段保持下文一致。

| 组别 | `vllm_omni/deploy/` 下 YAML | AR temperature | 请求 seed | 次数 |
| --- | --- | --- | --- | --- |
| non_pd_greedy | `hunyuan_image_3_moe_e2e_check.yaml` | 0 | 42 | 2 |
| pd_greedy | `hunyuan_image_3_moe_pd_e2e_check.yaml` | 0 | 42 | 2 |
| non_pd_rng | `hunyuan_image_3_moe_e2e_rng_check.yaml` | 1 | 43 | 2 |
| pd_rng | `hunyuan_image_3_moe_pd_e2e_rng_check.yaml` | 1 | 43 | 2 |

共 8 次请求。先完成贪心两组及对比，再完成随机两组。请求 seed=43 同时覆盖 AR/DiT
的 YAML 默认 seed=42，P 的有效 AR 参数由 D 派生，不能另设 P seed。
不要求 seed=42 和 seed=43 两组的输出彼此相同。

四份配置继承当前基础配置的 NPU 布局：P=4–7，D/非 PD AR=8–11，DiT=12–15。
沿用已有部署命令和环境，仅替换 `--stage-configs-path`（若原命令使用 `--deploy-config`，
则替换该参数，二者不要同时传入）。保留本轮 commit、实际启动命令、模型路径、vLLM/
vLLM-Omni/vLLM-Ascend/torch/CANN 版本及完整启动日志；不升级依赖或修改算子配置。
确认启动解析后的 TP、调度、采样、DiT 步数/CFG 设置符合上表。

以下命令在服务器 Bash 中执行。服务启动完成后，为当前组设置变量：

```bash
# 依次使用表格中组名及 seed；这里是第一组。
RUN=non_pd_greedy
SEED=42
mkdir -p "e2e-results/$RUN"
for REP in 1 2; do
  curl --fail-with-body --max-time 1800 \
    http://localhost:9000/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"generate a sleeping cat\",\"model\":\"/new_data/HunyuanImage-3.0-Instruct/\",\"use_system_prompt\":\"en_unified\",\"bot_task\":\"think\",\"num_inference_steps\":50,\"n\":1,\"seed\":$SEED}" \
    -o "e2e-results/$RUN/response_$REP.json" || break
done
```

AR temperature/max_tokens 使用 YAML；不要在图片请求中添加未经接口支持的 AR 参数。
沿用相同模型目录；若实际目录不同，四组统一替换。为每次部署分别保存完整服务日志，
包括全部 worker/rank 输出，例如 `non_pd_greedy.log`；不要只保存 curl 返回内容。
保留两次响应 JSON 及其中图片（若返回 URL，及时保存图像）。

## 日志覆盖与判定顺序

已有 `[HY3_EQ] ar` 打印完整 prompt/output token IDs、SHA256、CoT 文本、ratio 和目标尺寸；
本次补充 `finish_reason/stop_reason`。先确认自然停止、正常 ratio、输出未到 8192 上限。

新增 `[HY3_EQ] kv_full` 和 `[HY3_E2E]` 如下。每条日志带 request ID 和逻辑 TP rank，
不同部署按组别/第几次请求配对，再按 rank 0–3 配对；不要直接按 request ID 或时间戳比。

| 事件 | 检查项 |
| --- | --- |
| `kv_full event=send/receive` | AR→DiT 全部层 K/V 的 shape、dtype、完整字节 SHA256、整体摘要及 seq_len |
| `dit_request` | DiT 接收的完整 KV、实际 prompt/归一化 CoT/system prompt、seed、尺寸、guidance、步数 |
| `dit_initial` | 初始 latent 完整哈希及统计、generator 初始 seed、实际 timesteps/sigmas、scheduler 配置 |
| `dit_condition` | KV 复用长度、截断后 input IDs、mask/position 等张量、query/seq 长度 |
| `dit_injected_kv` | 真正注入每层 attention 的有效 KV，全部元素摘要 |
| `dit_step` | 第 1、10、25、50 步的 timestep、guidance 后 prediction、scheduler 更新后 latent 的完整摘要和统计 |
| `dit_final` | 去噪结束、VAE 缩放之前的最终 latent |
| `vae_output` | VAE 解码后、图像后处理之前的浮点图像张量 |
| `image` | 最终 PIL 图像的尺寸、mode、像素原始字节 SHA256 |

当前模型每个 rank 应有 32 层、64 个 K/V 张量，四个 rank 都要检查。
`kv_full` 观测的是 AR→DiT 传输，不是 Mooncake P→D 的内部传输。
同一次请求 AR send、DiT receive、dit_request 的完整 KV 应按相同 rank 对齐。
实际注入 KV 可能只取完整传入 KV 的前缀；因此跨 PD/非 PD 比较同种事件，
不要把完整传入 KV 与截短后的 `dit_injected_kv` 整体哈希直接比较。

验收分两层：先比较同配置第 1/2 次请求是否可重复，再比较 PD/非 PD 对应配置。
依次比较完整 AR IDs/CoT/ratio、传入及有效 KV、DiT 条件与初始 latent、去噪检查点、
最终 latent、VAE 输出及最终像素；所有浮点张量同时核对 shape/dtype。
这两层均逐项一致，才认定本轮覆盖范围内逐位等价通过。

- AR 先不同：回到完整 AR 生成定位，不归因于 DiT。
- AR 相同而传输/有效 KV 不同：先定位导出、传输和注入边界。
- KV/条件相同而初始 latent 不同：检查 DiT 有效 seed、generator 和形状。
- 初始 latent 相同而去噪开始不同：用检查点缩小差异区间，按需加密步数。
- 最终 latent 相同而图像不同：检查 VAE/后处理；服务端像素相同而客户端不同，检查返回编码链路。

不能把缺失日志当作一致。出现 `event=probe_error`、`failed to fingerprint KV`、
缺失 rank/步骤或 AR 截断，均需先解决，不能宣布等价通过。
这些探针在默认 `forward → _generate → HunyuanImage3Text2ImagePipeline.__call__` 路径运行；
无需开启 step execution。探针只读取张量，不清空/恢复 KV，也不消耗 generator 随机数。

## 保存与必要时的数值比较

初始和最终 latent 默认保存到服务器 `/tmp/hy3-e2e/non-pd/` 或 `/tmp/hy3-e2e/pd/`，
文件名包含 request ID、TP rank、事件名，日志 `event=artifact` 给出绝对路径。
按本轮四组分别归档这些文件和 request ID 对应关系，不要混入旧轮次文件。
文件内容是 tensor 字典，使用 `torch.load(path, map_location="cpu", weights_only=True)` 读取。
比较相同 rank 的 `latents`，先确认 shape/dtype；浮点误差用转换为 float32 后的差计算
`max_abs` 和 `mean_abs`。哈希不同但均值相同不构成数值等价。

图像应解码后比较尺寸、mode 和像素，而不是比较 JSON、base64 文本或 PNG/JPEG 文件哈希。
可用 Pillow 对响应中的图片解码，用 `hashlib.sha256(image.tobytes()).hexdigest()`
对照服务端 `image.pixel_sha256`；使用相同 mode，不额外缩放或有损重编码。

若出现微小浮点差异，本轮先标记“未通过逐位等价”，保留上述张量及图片；
结合同配置重复实验评估 NPU 数值波动，再约定 max_abs/mean_abs、像素 PSNR/SSIM 阈值。
不要凭视觉近似直接验收；本轮不预先放宽阈值。

全量 KV/张量日志和保存操作会产生同步、CPU 拷贝及 IO 开销，本轮耗时不用于性能结论。
通过后再推进 P 异步支持；性能实验需要关闭这些开关。
本轮不覆盖 CFG>1、图像编辑、其他尺寸/prompt、并发和多 P/D 拓扑。

## 插桩代码检查

在依赖已安装的测试环境中执行：

```bash
pytest --noconftest tests/utils/test_debug_fingerprint.py tests/utils/test_hunyuan_e2e_debug.py -q
```

新增测试使用真实方法体和 CPU 模拟模型/scheduler，覆盖普通 forward 路径的日志可达性、
开启/关闭探针后的输出与 RNG 状态一致、注入 KV 不被清空、完整摘要和 YAML 继承。
这些检查不替代上述 NPU 端到端实验。
