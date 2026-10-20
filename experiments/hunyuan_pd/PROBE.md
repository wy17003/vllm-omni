# 完整t2i批次客户端

完整方案见 [EXPERIMENT.md](EXPERIMENT.md)。推荐用run_experiment.py一次执行当前部署的全部点；
无需先做容量探测。probe_t2i_batch.py保留文件名，现支持正式实验的逐轮固定输入、现有日志TTFT/TPOT及E2E汇总。
只依赖Python标准库与curl，不加载模型、不启动服务、不修改运行中配置。

## 参数

- --batch-files：每轮一个冻结的catalog格式JSONL，恰好C行，检查固定参数、哈希和规范化重复文本。
- --rounds：1或3，默认1；正式用3，所有轮成功才选最后一轮。
- --deployment：pd/npd仅描述阶段拓扑，映射日志字段；请求始终为完整t2i。
- --server-log：本机可读的追加日志，需服务--log-stats，无无关请求、不轮转。
- --stage-max-num-seqs：PD传32 32 1，非PD传32 1；仅记录声明并追加目录标签，不配置服务器。
- --timeout：curl整个传输超时，默认180秒；正式方案显式统一1800秒，包含串行DiT排队预算。
- --base-url：默认http://localhost:9000。
- --model：默认/new_data/HunyuanImage-3.0-Instruct/。
- --output-dir：必须是新目录，拒绝覆盖；最终路径会打印。
- --prepare-only：只准备请求，不发送HTTP。

单点示例（在推理服务器仓库根目录执行；可将一行拆为shell续行）：

```bash
python experiments/hunyuan_pd/probe_t2i_batch.py --deployment pd --group long --concurrency 24 --rounds 3 --stage-max-num-seqs 32 32 1 --batch-files experiments/hunyuan_pd/schedule/pd-long-c24/round1.jsonl experiments/hunyuan_pd/schedule/pd-long-c24/round2.jsonl experiments/hunyuan_pd/schedule/pd-long-c24/round3.jsonl --server-log /home/w00934428/hy3_run01/pd-server.log --timeout 1800 --output-dir /home/w00934428/hy3_single01/pd-long-c24_pc-on_step-off
```

为兼容旧调用，不传batch-files仍从short/long池前缀选取，三轮复用同一批，
这不是本轮主题轮换方案。超过池容量会拒绝，不循环补齐。混合C必须为偶数。
正式方案优先使用run_experiment.py，它还会核对整个冻结清单的文件哈希。

## 输出

- plan.json：参数、输入文件哈希、配置声明、最终路径和执行状态。
- roundN_inputs/requests.json及request_bodies/：实际HTTP请求和来源元数据。
- roundN_inputs/batch.jsonl：C行，保留benchmark所需width/height；HTTP正文只发送size。
- roundN_responses/：各请求原始响应。
- roundN.json：逐请求response_latency_s、成功状态、实际尺寸、E2E/AR统计。
- roundN_server.log：传server-log时保存的日志字节片段。
- selected_result.json：所有轮成功后生成，本轮方案选第3轮。

E2E包含curl启动、HTTP和响应写盘，不含随后JSON/base64/PNG验证；client_wall_s包含验证，
不得除以N当平均E2E。只验证PNG头部尺寸，不判断画质。
TTFT取PD P或非PD AR的stage0；TPOT取PD D的stage1或非PD AR的stage0。
阶段本地TTFT不是完整PD请求TTFT。缺字段、全零或日志覆盖数不符时N/A，不填零。
错误/超时使该批无效，等本批其他请求结束后停止，不重试、不继续下一轮。
服务器可能在curl超时后继续工作，重新实验前应停止失败服务并确认清理。
prefix caching必须开启，本轮step_execution=false；客户端不能从图片响应验证服务配置生效。
