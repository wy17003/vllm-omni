# HunyuanImage3 PD 简化性能实验

> 本文保留为 run01 原方案。当前补充实验使用 [EXPERIMENT_RUN02.md](EXPERIMENT_RUN02.md)：
> 复用 run01 PD short/mixed，补跑 PD long，执行修订后的 NPD 六个点。

本轮直接执行完整t2i，用现有阶段日志统计TTFT/TPOT，用客户端逐请求响应时间统计E2E。
不增加AR-only模式，不搜索容量，不要求先跑探针，也不要求加载tokenizer校准输入。
脚本只向已启动服务发送请求；服务由用户通过vllm serve启动。

## 1. 实验矩阵

| 对比口径 | PD | 非PD | concurrency/card |
|---|---|---|---|
| AR子系统卡数归一 | C24，P4+D4卡 | C12，AR4卡 | 3 / 3 |
| 完整服务卡数归一 | C24，共12卡 | C16，共8卡 | 2 / 2 |

复用同一个PD C24结果。所有点都执行完整t2i，都会记录AR指标和E2E。
AR口径仍受完整管线调度和DiT影响，不是隔离的AR实验。
请求总数不同，采用对应子集及近似分布匹配，不声称完全相同负载。

各配置分别运行短、长、长短各半三组；每点三轮，每轮N=C，并发一次发出。
共9个点、27批、468请求：PD216个，非PD252个。第三轮为主结果，保留前两轮，
不取最好一轮，不把不同输入的三轮当成相同负载的重复测量计算稳定性。
这是有限突发批次实验，不测持续饱和吞吐；批次耗时/N不是平均请求E2E。

## 2. 配置

直接使用configs/pd.yaml和configs/npd.yaml，已展开NPU参数，不依赖平台overlay。
基础部署YAML保持原样。两边同一代码、模型、设备类型和客户端环境，TP均为4。

| 设置 | PD | 非PD |
|---|---|---|
| 阶段/卡号 | P:0–3；D:4–7；DiT:8–11 | AR:4–7；DiT:8–11 |
| max_num_seqs | 32 / 32 / 1 | 32 / 1 |
| enable_prefix_caching | true | true |
| DiT step_execution | false | false |
| DiT attention | TORCH_SDPA | TORCH_SDPA |
| AR max_num_batched_tokens | 8192 | 8192 |
| AR / DiT memory utilization | 0.8 / 0.65 | 0.8 / 0.65 |

max_num_seqs=32覆盖最大C24，避免该上限主动限制批次；它不是客户端并发，
也不是预留32请求显存，长输入/输出仍可能导致KV不足、抢占或OOM。
DiT取1与step_execution=false一致：HTTP仍并发，DiT按请求执行，E2E包含排队。
DiT可能成为瓶颈，因此AR加速不保证E2E同比改善。

prefix caching保持开启，可复用system prompt或曾请求过的输入，也有缓存管理/hidden-state复制成本。
本分支支持相应配置，但本机尚未做真实NPU缓存命中验证。不能把所有收益归因于PD。
此前step路径在prepare_encode时缺少Omni forward context，本轮关闭step执行，
暂不将该兼容性修复与性能实验混在一起；不是说模型没有声明step支持。

配置关闭AR逐token等价调试和KV fingerprint，保留普通统计日志。
启动前清理此前手工设置的Hunyuan RNG/E2E/KV调试变量、设备可见性重映射及attention覆盖，
避免覆盖配置。若需修改卡号，两边须保持硬件/拓扑可比并记录。

固定请求：1024x1024、n=1、50步、seed=42、bot_task=think、use_system_prompt=en_unified、
guidance_scale=0、response_format=b64_json。验证返回PNG头部尺寸，不做resize，不等同画质检查。
显式size会影响AR停止语义，不能与不传size的历史成绩直接混比。

配套源码修复显式guidance_scale（含0）的provided标记：HTTP赋值与YAML默认值保留显式语义，
防止Hunyuan回退至默认CFG=5。必须把源码修改一起同步至推理环境，不能只复制脚本。
本次不修改step forward context代码。

## 3. 输入已准备完成

直接使用schedule/manifest.json及27个每轮JSONL，无需token校准或再次选prompt。
启动前验证文件SHA256；批内禁止重复，三轮及PD/非PD允许复用，每轮轮换部分主题。

- 基础池：32主题，各一条短版和长版，共64条不同文本；每组25中文、7英文。
- 每批英文25%、中文75%；混合组严格50%短、50%长。
- PD按语言、同语言内字符长度分层选24条；非PD选其12/16条子集，近似匹配字符均值/中位数。
- 混合C12仅3条英文，短/长子组分别1/2条，轮次间交换；子组不能都精确25%。
- manifest记录每轮主题ID、字符分层统计、输入哈希、与此前轮次的重合数。

字符数只是选样依据，不当作token数。短/长是现有相对分类，不称作2K/4K/8K桶。
当前非PD相对PD的整批平均字符数偏差绝对值：短组≤0.3%，长组≤2.2%，混合组≤9.6%；
小样本英文子层可到16.3%。没有保证完整输入token分布或AR输出长度完全一致。
完整输入含system prompt，“长”组也不保证足够长到体现PD收益；可事后查看已有日志的输入长度，
但不因性能成绩反向改输入。inputs/calibrate_tokens.py仅保留为可选工具。

数量足够，无需补prompt。若要求纯组C24三轮完全不重合，每组需72条、目前缺40条，
本轮没有此要求。纯组PD相邻轮更换8条，混合组轮换更多主题，详见manifest。

轮换不清空prefix cache；不同部署和C的历史暴露比例、淘汰状态无法完全一致。
例如纯组PD第三轮全部已在前两轮出现，而非PD可能更少。因此报告应写
“prefix caching开启，固定顺序、轮换输入，第3轮”，不写“同一批输入的第三次重复”。
同一服务不同组/并发间也保留缓存。当前reset_prefix_cache路径不执行实际清理，不调用它伪造冷缓存。
这项简化实验不能单独分离缓存收益；保留所有轮次和固定执行顺序以便解释。

只有修改源池才需重建双方清单；使用新目录，拒绝覆盖旧清单：

```bash
python experiments/hunyuan_pd/prepare_schedule.py --output-dir /home/w00934428/hy3_schedule_v2
```

## 4. 执行：先完成所有PD，再启动非PD

以下在推理服务器仓库根目录运行。先同步全部源码修改和schedule，确认运行环境加载修改后的代码。
两个服务顺序启动，不同时运行。服务ready后直接实验，不另加容量探针或业务请求预热。
禁止其他客户端访问同一服务。日志必须是客户端本机可读的追加文件，不轮转、不截断。

**终端A：启动PD，等待ready**

```bash
mkdir -p /home/w00934428/hy3_run01
vllm serve /new_data/HunyuanImage-3.0-Instruct/ --omni \
  --port 9000 --stage-configs-path experiments/hunyuan_pd/configs/pd.yaml \
  --log-stats > /home/w00934428/hy3_run01/pd-server.log 2>&1
```

**终端B：执行PD全部点**

```bash
python experiments/hunyuan_pd/run_experiment.py \
  --deployment pd --output-root /home/w00934428/hy3_run01 \
  --server-log /home/w00934428/hy3_run01/pd-server.log --timeout 1800
```

依次short C24三轮、long C24三轮、mixed C24三轮。完成后在终端A停止PD，
确认进程退出、卡和端口释放，再启动非PD。

**终端A：启动非PD，等待ready**

```bash
vllm serve /new_data/HunyuanImage-3.0-Instruct/ --omni \
  --port 9000 --stage-configs-path experiments/hunyuan_pd/configs/npd.yaml \
  --log-stats > /home/w00934428/hy3_run01/npd-server.log 2>&1
```

**终端B：执行非PD全部点**

```bash
python experiments/hunyuan_pd/run_experiment.py \
  --deployment npd --output-root /home/w00934428/hy3_run01 \
  --server-log /home/w00934428/hy3_run01/npd-server.log --timeout 1800
```

先C12的short/long/mixed各三轮，再C16的short/long/mixed各三轮。每批全部完成才进入下一批。
正式命令预先统一curl总超时1800秒，为串行DiT排队留预算，不是预测每个请求都需1800秒。
脚本默认兼容原curl的180秒；本方案显式传1800，两边必须一致。

可选--prepare-only只生成执行计划和请求，无需服务或server-log；使用独立新目录，
如--output-root /home/w00934428/hy3_dryrun。正式运行不能复用已创建的dryrun目录。

## 5. 结果与指标口径

各部署summary.json汇总各点第三轮；点目录包含组、C、pc-on、step-off和各阶段上限，
例如pd-long-c24_pc-on_step-off_s0-32_s1-32_s2-1。
保存plan、每轮实际请求、响应、逐请求结果、日志片段、selected_result。
部署目录另存Git HEAD/status/diff、声明的配置和manifest；未跟踪源码须自行完整同步。
配置副本和max_num_seqs仅是客户端声明，不是运行时读回，须核对服务启动及CLI/环境覆盖。

| 指标 | 来源 | 单位/语义 |
|---|---|---|
| TTFT | PD stage0或非PD stage0的vllm_ttft_ms | ms，阶段本地首token时间 |
| TPOT | PD stage1或非PD stage0的vllm_tpot_ms | ms/token，阶段本地输出token间隔统计 |
| E2E | 每请求response_latency_s | s，curl启动至响应写盘，含传输，不含随后PNG校验 |

每项输出样本数、均值、P50、最大值。PD的P阶段TTFT不含完整P→D链路等待，
不是统一的请求级TTFT，不能把P与D的TTFT相加。非流式图像HTTP响应也无法直接测AR首token。
OmniTiming total是另一种服务器时间口径，不代替客户端E2E。

TTFT/TPOT读取现有StageRequestStats表，不新增服务器埋点。每轮记录服务日志起止字节区间，
要求无其他流量且日志正常刷新。请求表数需恰好N，每个指标都存在且为正，否则该项N/A，附observed_n。
全零被日志省略不是“测得零”。N/A时查看roundN_server.log及完整日志，不用部分样本冒充整轮成绩。
日志格式或刷新差异可能影响采集，真实运行时需核验。

先按AR卡数口径比较PD24/非PD12的TTFT、TPOT，再按总卡数口径比较PD24/非PD16的E2E；
两种对比都可查看其他两项指标，勿混淆归一口径。
长输入、多请求可能让P/D隔离减少prefill/decode干扰，但KV传输、prefix命中和DiT排队会影响结果，
不能预先承诺加速。

## 6. 失败与验证范围

HTTP500、curl超时、响应缺失、图片数量或实际尺寸不符均使该批无效。
等待本批其他请求结束后停止后续轮次/点，保留响应和日志，不生成失败点selected_result，
不自动重试、不降并发、不关闭prefix caching。失败点不进入正式对比。
客户端超时不保证服务器停止；重试前必须停止失败服务并确认在途请求清理。
修改代码、上限或超时后作为新实验版本重新运行双方，用新输出目录，勿只重跑有利结果。

本地验证覆盖清单确定性、比例/唯一性/子集关系、真实curl调用模拟HTTP服务的三轮输入轮换、
已有日志解析和HTTP失败停止。尚未执行真实NPU推理，不能据此认定容量、缓存命中或模型出图已验证。
核心guidance测试在tests/entrypoints/openai_api/test_hunyuan_image3_seed.py，
需要装有vllm/torch/pytest的环境；本地缺少这些依赖。
