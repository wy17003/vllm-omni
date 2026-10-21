# HunyuanImage3 run03：AR 高并发与两波请求实验

本轮考察 PD 的 AR 流水与 prefill/decode 干扰隔离收益。所有请求仍执行完整 t2i，
不增加 AR-only 服务模式，不实现不同 AR KV 长度的 DiT 合批。
每个实验点只执行一次；统一 `max_num_batched_tokens=8192`，不做容量搜索或自动降并发。
旧 run01/run02 脚本、配置及结果保留，新入口为 `run_ar_experiment.py`。

## 1. 固定实验矩阵

| 执行顺序 | 部署 | 负载 | t=0 请求数 | t=5s 新请求数 | 总请求 / HTTP 最大在途数 |
|---|---|---|---:|---:|---:|
| 1 | PD | single，单批高并发 | 48 | 0 | 48 / 48 |
| 2 | PD | waves，两波请求 | 24 | 24 | 48 / 48 |
| 3 | NPD | single，单批高并发 | 24 | 0 | 24 / 24 |
| 4 | NPD | waves，两波请求 | 12 | 12 | 24 / 24 |

PD 使用 P4+D4，共 8 张 AR 卡；NPD 使用 4 张 AR 卡。
单批请求数/AR卡均为 6；两波每波请求数/AR卡均为 3，两波都未完成时最多为 6。
两边各另有 4 张 DiT 卡，总卡数分别为 12/8。本轮只做 AR 卡数归一比较。
PD/NPD 请求量不同，因此同时报告原始指标和 output tokens/s/AR卡，不能称作等总负载比较。

共 144 个正式请求。每个部署新启动服务后，脚本先执行 1 个相同的独立预热请求，
检查 HTTP 成功与实际 1024×1024，再依次执行 single、waves。
预热不计入正式结果，正式 prompt 不参与预热；共计 146 次 t2i。
每点等待全部图像返回后再进入下一点，但**同一点的第二波不等第一波图像返回**。

两波的 5 秒间隔是双方一致、事先固定的到达计划，不根据某一部署完成速度触发。
程序记录计划与实际发起时间；偏差超过 0.5 秒标为不可用于正式比较，保存结果后停止。
该时间是启动 curl 前的主机时间，网络接收与服务开始时间也可能晚于它，应结合 request_start 检查。

## 2. 服务和请求配置

| 项目 | PD | NPD |
|---|---|---|
| YAML | configs/run03_pd.yaml | configs/run03_npd.yaml |
| 设备 | P:0–3；D:4–7；DiT:8–11 | AR:4–7；DiT:8–11 |
| TP | 每阶段 4 | 每阶段 4 |
| max_num_seqs，按阶段 | 64 / 64 / 1 | 64 / 1 |
| AR max_num_batched_tokens | 8192 / 8192 | 8192 |
| enable_prefix_caching / enable_chunked_prefill | true / true | true / true |
| DiT step_execution | false | false |
| DiT attention | TORCH_SDPA | TORCH_SDPA |
| AR / DiT memory utilization | 0.8 / 0.65 | 0.8 / 0.65 |

`max_num_seqs=64` 覆盖最多 48 个正式在途请求，避免人为把两波串行化；它不保证 KV 容量足够。
`8192` 是每次调度的 token 预算，不是请求总长度，也不是请求数。
这些输入约 1.9K 完整 tokens，独立预热后公共 system prefix 可命中缓存，实际待计算长度会更短。
即使输入长度相近，24/48 个新请求的总待计算量也可能跨多个调度步。
维持 8192 保留原配置的吞吐条件；不通过缩小预算人为延长 P 阶段来制造流水窗口。
是否出现流水、第二波是否打断 NPD decode，必须以本轮事件和缓存命中记录判断，不能由预算值保证。

固定 HTTP 参数：`size=1024x1024`、`n=1`、`num_inference_steps=50`、`guidance_scale=0`、
`seed=42`、`bot_task=think`、`use_system_prompt=en_unified`、`response_format=b64_json`。
AR temperature=0，max_tokens=8192。实际响应必须为一张 PNG，IHDR 尺寸为 1024×1024。
这只是尺寸检查，不是画质或数值等价性检查。

DiT 仍串行，因此 E2E 仅作为完整请求诊断，不以本轮 E2E 论证 DiT 批处理收益。
并发增加会增加 DiT 排队及 AR→DiT KV 保留压力，AR 完成时间仍可能包含这部分耦合影响。

## 3. 输入已冻结，无需手工选样或在线 token 校准

直接使用 `schedule_run03/manifest.json` 和四份 JSONL；`prompts.md` 可读览全部文本。

- 96 条唯一正式 prompt，中文 72 条、英文 24 条；单批与两波使用不同主题集合。
- 每批及每波中文 75%、英文 25%。同一主题含三种中文视点及一种英文描述，文本开头即有差异。
- 单批 48 条、两波另 48 条；两波内部 24+24 无重复，均不复用单批文本。
- NPD 是对应 PD 批次/波次的 1/2 子集，按语言、完整输入 token 长度匹配；两边使用同一份冻结清单。
- 完整输入为 **1904–1993 tokens**，已用提供的 tokenizer 和当前 t2i/think 模板离线计数，包含 system prompt。
  PD/NPD 的 single 均值为 1929.98/1930.21；waves 均值为 1928.33/1927.67。

模板化拍摄要求位于不同场景描述之后，避免所有 user prompt 共享很长开头。
这不消除公共 system prefix，也不保证双方缓存命中数或 AR 输出长度相同；结果会记录实际值。
输入仍保留 `group=long` 以复用原请求加载器，该字段不代表本轮另有短/长实验分组。

源文本与构造程序为 `prepare_run03.py`，不需要加载模型权重。
tokenizer SHA256 为 `c2439418b9be76a54d8b5ff1ed89b37e36ec1735730f7da99b5aabb83a73db64`。
运行时必须使用相同 tokenizer，脚本逐请求检查日志中的完整输入长度与冻结值。
已为本目录添加 Git JSON/JSONL 例外和 LF 属性，避免清单漏传及 Windows/Linux 换行哈希不一致。
manifest 中的 Windows tokenizer 路径仅为生成来源；实际运行使用 `--tokenizer-json`，按哈希核对。

若需要在 Linux 从源码重新构造，双方共用下述新目录，并同时修改双方 `--schedule-dir`：

```bash
cd /home/w00934428/projects/vllm-omni || exit 1
python experiments/hunyuan_pd/prepare_run03.py \
  --tokenizer-json /new_data/HunyuanImage-3.0-Instruct/tokenizer.json \
  --output-dir /home/w00934428/hy3_run03_schedule_linux
```

默认应直接使用仓库冻结清单。重建不是运行前必需步骤，且禁止覆盖已有目录。

## 4. 完整执行步骤：先全部 PD，再全部 NPD

先把本次全部源码、配置、脚本、冻结清单同步到服务器；不能仅复制 YAML 或客户端脚本。
确保推理环境加载当前 checkout 的 vllm_omni，而非另一份已安装源码。
使用同一台 Linux 主机启动服务和客户端；记录依赖、设备类型、拓扑，双方保持一致。
沿用此前已验证的 NPU 环境，去掉此前手工添加的 debug、RNG、KV fingerprint、attention 及设备可见性覆盖。
普通 HY3_EQ 日志保留用于 token 审计，AR 逐 token 等价调试保持关闭。
服务只接受本实验流量；日志保持追加，不轮转、不截断。

**终端 A：启动 PD，等待服务 ready。**

```bash
cd /home/w00934428/projects/vllm-omni || exit 1
mkdir -p /home/w00934428/hy3_run03
VLLM_OMNI_HY3_AR_TRACE=1 vllm serve /new_data/HunyuanImage-3.0-Instruct/ --omni \
  --port 9000 \
  --stage-configs-path /home/w00934428/projects/vllm-omni/experiments/hunyuan_pd/configs/run03_pd.yaml \
  --log-stats > /home/w00934428/hy3_run03/pd-server.log 2>&1
```

`VLLM_OMNI_HY3_AR_TRACE=1` 必须在启动命令设置，使 orchestrator、scheduler 和 worker 均继承。
仅在 YAML 某个 stage 设置不足以覆盖全部观测点。默认不开启该日志。

**终端 B：执行 PD 两个点。**

```bash
cd /home/w00934428/projects/vllm-omni || exit 1
python experiments/hunyuan_pd/run_ar_experiment.py \
  --deployment pd \
  --schedule-dir experiments/hunyuan_pd/schedule_run03 \
  --tokenizer-json /new_data/HunyuanImage-3.0-Instruct/tokenizer.json \
  --output-root /home/w00934428/hy3_run03 \
  --server-log /home/w00934428/hy3_run03/pd-server.log \
  --timeout 3600
```

依次为独立预热、single 48、waves 24+24，各正式点一轮。
全部成功后，在终端 A 停止 PD，确认相关进程退出、设备及端口释放。

**终端 A：启动 NPD，等待服务 ready。**

```bash
cd /home/w00934428/projects/vllm-omni || exit 1
VLLM_OMNI_HY3_AR_TRACE=1 vllm serve /new_data/HunyuanImage-3.0-Instruct/ --omni \
  --port 9000 \
  --stage-configs-path /home/w00934428/projects/vllm-omni/experiments/hunyuan_pd/configs/run03_npd.yaml \
  --log-stats > /home/w00934428/hy3_run03/npd-server.log 2>&1
```

**终端 B：执行 NPD 两个点。**

```bash
cd /home/w00934428/projects/vllm-omni || exit 1
python experiments/hunyuan_pd/run_ar_experiment.py \
  --deployment npd \
  --schedule-dir experiments/hunyuan_pd/schedule_run03 \
  --tokenizer-json /new_data/HunyuanImage-3.0-Instruct/tokenizer.json \
  --output-root /home/w00934428/hy3_run03 \
  --server-log /home/w00934428/hy3_run03/npd-server.log \
  --timeout 3600
```

依次为同一独立预热、single 24、waves 12+12。双方统一 3600 秒请求超时，为串行 DiT 留出预算，
不表示预计 AR 要执行一小时。若仍超时，保留失败结果，后续另行决定是否调整双方超时。
不插入额外业务预热，不调用无效的 prefix cache reset；新部署从新服务开始，各点间保留缓存。

可选：只生成计划和实际请求，使用独立的新输出目录，不访问服务：

```bash
python experiments/hunyuan_pd/run_ar_experiment.py \
  --deployment pd --prepare-only \
  --schedule-dir experiments/hunyuan_pd/schedule_run03 \
  --tokenizer-json /new_data/HunyuanImage-3.0-Instruct/tokenizer.json \
  --output-root /home/w00934428/hy3_run03_dryrun
```

脚本拒绝覆盖已有部署目录。正式运行不可复用 dryrun 输出目录；也不可删除失败数据后原地重跑冒充首次执行。

## 5. 观测指标与边界

沿用 `round1.json.ar_metrics` 的原生 TTFT/TPOT：PD TTFT 为 P 阶段首 token，TPOT 为 D 阶段；
NPD 二者均为 stage0。PD 原生 TTFT 不包含 P→D 等待，因此另看新增的 D 首次输出和 AR 完成时间。
客户端逐请求 E2E 保留在 `records`，汇总为 `e2e_s`。

新增 `ar_trace` 使用同主机跨进程 `monotonic_ns`，按实际 token 日志匹配请求：

| 事件/字段 | 含义 |
|---|---|
| request_start | orchestrator 接收请求、即将提交 stage0 |
| admit | scheduler 接收请求；记录内部/外部 ID、实际 max_num_seqs、token 预算和 prefix caching |
| forward_submit | NPU TP rank0 即将调用模型 forward，主机侧提交时间 |
| outputs | scheduler 收到模型输出，记录本次与累计 token 数；不记录 token 文本 |
| kv_ready | D scheduler 确认完整 prompt KV 已接收，可继续调度 |
| stage_done | orchestrator 收到阶段完成；P 完成后才进行跨阶段提交 |
| prefill_first_token_mono_ns | P/AR 首次 token 结果到达 scheduler |
| prefill_done_mono_ns | PD 为 P stage_done；NPD 为首 token 到达 scheduler |
| ar_complete_mono_ns / ar_complete_s | 完整 AR stage_done 时间 / 相对 request_start 时延 |
| cached_tokens / uncached_tokens | P/NPD AR 首次 prefill 的本地 prefix 命中 / 未命中输入数 |

这些时间含主机调度与通信，不是设备 kernel 起止或纯 KV 传输耗时。
P 首次 forward 到首 token 是主机观测跨度；P stage_done 到 D KV ready 包括提交、排队与传输。
不通过设备同步改变执行行为；证明 kernel 级重叠仍需 profiler，本轮不增加该步骤。
逐步事件日志有一定开销，双方都开启相同观测配置，不能与无 trace 的旧结果作严格性能配对。

结果自动提供：

- 每请求 AR 完成时间及按波次的 mean/p50/p95/max；首 forward、P 首 token、P 完成、D KV ready、D 首次 forward/输出。
- AR 总窗口：最早客户端发起至最后 AR 完成；输出 tokens/s 及除以 8/4 的 tokens/s/AR卡。
  PD 的 P 首 token 会在 D 输出中携带，输出总数只计一次。该指标是有限批次吞吐，不是稳态饱和吞吐。
- 两波中第一波在第二波实际发起时仍产生后续 tokens 的请求数，及前后各 5 秒的 token 输出间隔。
  跨越第二波边界的间隔按结束时刻归入后窗，不丢弃可能被 prefill 拉长的间隔；保留逐请求和整体统计。
- PD 最早 D 新 token 到达时，已经进入系统但尚未结束 P 的请求数，以及已提交 P forward 未结束的请求数。
  尚未发起的第二波不会被误计为并行请求。
- 实际缓存命中/未命中 token 数，用于判断缓存是否明显削弱 prefill 工作量。

若 `two_wave_overlap.target_window_observed=false`，说明第二波未落在已观测到的第一波 decode 窗口内。
数据可保留，但不能据此判定“PD 没有隔离收益”；先看实际到达/生成时间，再决定是否另做双方一致的新间隔。
不自动调间隔或挑最好结果。每点只有一轮，不报告重复实验稳定性或置信区间。

## 6. 失败处理与材料归档

继续采用 `warn_equal_length_v1`：等长 token ID 差异记警告，允许本轮实验继续，
但不宣称 KV/图像严格等价；长度差异、解码文本异常或日志覆盖不全仍阻断。
新 prompt 的实际 AR 输出尚未经 NPU 推理验证，因此仍有触发已知 round-trip 问题的可能。

HTTP/尺寸/token 审计失败，或 trace 缺失、配置不符、异主机时钟、事件不完整、发起时间偏差过大时，
保存请求、响应、逐请求结果、日志片段及失败原因后停止，不自动换 prompt、降并发或重试。
原始 TTFT/TPOT 缺失同样不能生成合格的 selected_result。
失败的 trace 不填写为 0；其 AR 总窗口和吞吐留空，避免误作有效结果。
如只是 trace 解析问题，已有完整原始日志可用于修正离线解析，无需立即重跑推理。

每个部署目录保存 manifest、声明 YAML、Git HEAD/status/diff、源文件 SHA256 和包版本。
每点目录名包括部署、single/waves、总请求数、b8192、pc-on、step-off 和各阶段上限。
点内保存 plan、round1_inputs/requests.json、实际 HTTP body、round1_responses、round1_server.log、
round1.json、合格时的 selected_result；部署 summary.json 汇总已执行点。
AR 上限、预算和 prefix caching 从运行中 scheduler 读回核对；设备和 DiT 参数仍须结合启动日志确认。

实验结束提供完整 `hy3_run03/pd`、`hy3_run03/npd` 与双方完整 server.log；大体积图像响应可暂不传，
但原地保留以备复核。现有 run01/run02 作为背景观察，不混入本轮四点的配对统计。

本地验证覆盖冻结清单、实际 tokenizer 重建、curl 两波并发、时间线解析及失败门禁。
本机无 NPU 推理环境，尚未验证真实设备容量、运行时事件覆盖和性能收益。
