# HunyuanImage-3.0：现有配置端到端等价验证

本轮只验证 NPU、1P1D、每阶段 TP=4、单请求 T2I（think）、CFG 关闭的完整链路。
保持 P `async_scheduling: false`，D 和非 PD AR `async_scheduling: true`；
保持 `step_execution: false`、KV 异步预取开启、prefix caching 关闭。
DiT YAML 同时设置 `guidance_scale: 0` 和 `guidance_scale_provided: true`。
后者区分“显式关闭 CFG”和通用采样参数的零值哨兵；需使用本轮修复后的
`OmniDiffusionRequest.__post_init__`，确保初始化不覆盖显式零值。
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
| `dit_request` | DiT 接收的完整 KV、实际 prompt/归一化 CoT/system prompt、seed、尺寸、ratio、有效/采样 guidance 及 provided 标记、步数 |
| `dit_initial` | 初始 latent 完整哈希及统计、generator 初始 seed、实际 timesteps/sigmas、scheduler 配置 |
| `dit_condition` | KV 复用长度、截断后 input IDs、mask/position 等张量、query/seq 长度，以及实际 TP/SP/CFG 大小、CFG 开启/并行状态 |
| `dit_injected_kv` | 真正注入每层 attention 的有效 KV，全部元素摘要；本轮应只有 `branch=positive, branch_count=1` |
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
pytest --noconftest tests/diffusion/test_diffusion_request.py tests/utils/test_debug_fingerprint.py tests/utils/test_hunyuan_e2e_debug.py -q
```

新增测试使用真实方法体和 CPU 模拟模型/scheduler，覆盖普通 forward 路径的日志可达性、
开启/关闭探针后的输出与 RNG 状态一致、注入 KV 不被清空、完整摘要和 YAML 继承。
这些检查不替代上述 NPU 端到端实验。

## err2.log 后的复跑

该请求正常返回 HTTP 200，AR 自然停止于 900 tokens、ratio_index=13，目标高×宽为
832×1216。四个 TP rank 的 AR→DiT 全层 KV send/receive/dit_request 摘要一致，
有效传输长度 2135=1236+900−1；这份日志未显示该边界传输损坏。
但 DiT 实际 guidance=5、尺寸为 1024×1024，不能计为原计划中 CFG 关闭的有效样本。

四次 `dit_injected_kv` 报错的直接原因是探针假设每层仅一个分支，实际 CFG 负向 prefill
把 `_injected_ar_kv` 扩展为正/负两个分支。修复后探针按真实 CFG 模式分别记录，
仍会对缺失 KV 或分支数不符报错；并未删除检查或吞掉缺失数据。
之前 CPU 路径测试直接构造 provided=true 的采样参数，未覆盖 YAML 到请求初始化的
零值处理；现已补充真实初始化方法和 T2I/IT2I 预处理回归检查。

更新代码及 YAML 后，先重启 PD 贪心配置并用原 seed=42 请求跑一次，核对：

1. `dit_request` 的 `guidance_scale=0`、`sampling_guidance_scale=0`、`guidance_scale_provided=true`。
2. DiT 尺寸等于本次 AR bridge 的目标尺寸；若 AR 仍为上述 900 tokens/ratio=13，应为高 832、宽 1216。
3. 四个 rank 都有 `dit_condition`，其中 TP=4、SP=1、CFG world size=1、`cfg_enabled=false`；
   四个 rank 都有完整 `dit_injected_kv` 正分支日志，且无 `probe_error`。
4. 50 步完成并正常返回图片；日志图片尺寸和客户端解码尺寸都应与目标一致。

该次成功样本可直接计为 PD 贪心组第 1 次，再连续请求第 2 次；随后执行非 PD 贪心两次并比较。
贪心两组通过后，继续原随机两组，各两次；不需要重新执行 AR 边界实验。
旧 err2.log 保留为诊断记录，不与修复后样本混用。

err2.log 中初始 latent 四个 rank 相同，但第 1 步起 rank 0/1 与 rank 2/3 的 prediction/
latent 已分成两组，最终像素也不同。现有日志不足以确定原因，更不能据此认定 PD/非 PD
不同。复跑时核对上述实际并行信息；若 CFG 关闭后仍有这种分组，先保存该次四 rank
完整日志与 latent 文件定位，不进入随机和性能实验。
