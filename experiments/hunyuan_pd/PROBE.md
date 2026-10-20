# 并发可完成性检查

`probe_t2i_batch.py` 对已启动并完成模型加载的完整 t2i 服务默认执行 N=C 的一个批次。
可完成性初查用 `--rounds 1` 即可，不必每次执行三轮。可用 `--rounds 3` 验证重复请求后的缓存路径，
三轮复用相同输入且选中第三轮；这仍是探针，不替代正式性能采集。每批内文本唯一。
任意一轮失败或实际图片尺寸不符即停止，不生成选中结果，已执行轮次的响应保留。

## 对照给定 curl

默认使用 `http://localhost:9000/v1/images/generations`、模型 `/new_data/HunyuanImage-3.0-Instruct/`，
以及 `use_system_prompt=en_unified`、`bot_task=think`、50步、n=1、seed=42。
每个请求通过独立 curl 进程提交，`--max-time 180` 是整个传输的超时（不是socket单次读取超时）。
可用 `--timeout` 覆盖；对比双方必须一致，超时记失败，不自动重试或提高时限后挑选成绩。
正文由UTF-8文件传给 `--data-binary`，不拼接shell命令。每请求原始JSON响应保存到对应轮次目录。

相对示例额外固定三个参数：

- `size=1024x1024`：满足已确定的同一实际尺寸约束。显式尺寸也改变think路径的AR停止标记，
  因此不能将本实验直接与未指定size的历史成绩混比；PD和非PD都走同一显式尺寸路径。
- `guidance_scale=0.0`：沿用此前实验参数及当前两份基础部署的DiT默认值。
- `response_format=b64_json`：用返回PNG的IHDR验证实际尺寸。只验证头部尺寸，不代替图像质量检查。

## 输入与调用

短、长各32条不同文本；混合按同主题短长交替，各占一半，要求C为偶数。
纯组取前C条，混合取前C/2对。禁止循环、复制或通过换seed扩充数量，检测规范化后的重复文本。
超过可用数量时在发送请求前报错。当前支持纯组C≤32、混合偶数C≤64。
不同并发的PD/非PD采用同一有序池的前缀；请求数不同时不可能拥有完全相同的整批请求。
它保证共享样本一致，但不能自动保证不同前缀的token分布相同，需在token校准后审核。

在推理服务器仓库根目录运行（需要Python与curl）：

```bash
python experiments/hunyuan_pd/probe_t2i_batch.py \
  --group long --concurrency 24 --rounds 1 \
  --stage-max-num-seqs 32 32 2 \
  --output-dir /home/w00934428/hunyuan_pd_results/pd-long-c24_pc-on_step-on
```

`--base-url`、`--model` 可覆盖默认值。加 `--prepare-only` 只生成输入，不访问服务。
必须选择新输出目录，已有结果不会覆盖。输出包括：

- `requests.json`、`request_bodies/`：确切请求及来源行、prompt哈希。
- `batch.jsonl`：恰好C行，含供CustomDataset使用的width/height；元数据不发送到服务。
  正式benchmark须显式设置 `--num-prompts C`，否则其循环取样逻辑可能重复输入。
- `plan.json`：输入唯一数、超时、执行状态、配置轮数。
- `round1/2/3.json`、`round1/2/3_responses/`：逐请求状态、响应时间、实际尺寸、原始响应。
- `selected_result.json`：仅指定轮次全部成功后生成，选中最后一轮。

`--stage-max-num-seqs` 按stage_id顺序声明上限：非PD传 `32 2`，PD传 `32 32 2`。
脚本将其记录到 `plan.json` 的 `server_config_declaration.max_num_seqs_by_stage`，并打印到终端，
同时给输出目录追加 `_s0-32_s1-32_s2-2`（PD示例）。上面命令的实际目录为
`pd-long-c24_pc-on_step-on_s0-32_s1-32_s2-2`；非PD可用
`npd-long-c16_pc-on_step-on_s0-32_s1-2`。每次重跑可在用户提供的目录名加入run01/run02。
`pc-on` 与 `step-on` 是用户填写的标签，分别代表prefix caching和DiT step execution开启。
**声明和目录名不是运行时验证**：这些参数不发送给服务，也不修改YAML；必须与实际启动配置一致。
未传阶段上限时，plan记录null，目录不追加配置标签，保留旧命令兼容性。
原探针只记录客户端C，无法用它反推出服务max_num_seqs；二者必须分别记录。
服务代码有DEBUG级stage_configs输出和DiT上限被降至1的WARNING，但不能依赖默认日志完整记录所有阶段。
同时保留服务启动命令、使用的YAML/继承配置及服务日志，以确认平台和CLI覆盖后的实际值。

## 服务配置与指标边界

最新实验要求：**禁止关闭prefix caching**。当前基础YAML仍是false，启动实验服务前必须在所用配置中
设 `enable_prefix_caching: true`，并核对非PD的AR及PD的P、D最终生效值；不要被CLI或其他overlay覆盖。
该选项针对AR的token KV前缀缓存，不等同于DiT计算缓存，也不等同于P→D或AR→DiT的KV传输。
脚本不修改部署YAML或运行中服务，不能从图片响应验证是否开启。

启用缓存后，批内prompt不重复也可能共享system prompt前缀；三轮复用输入还可能命中完整输入前缀，
具体取决于缓存容量、淘汰与实现。因此第三轮正式成绩须标为“prefix caching开启、重复输入暖缓存条件”，
不能只归因于PD分离。应记录缓存命中及实际prefill工作量。
探针也会预热缓存；正式实验两种部署应采用一致的启动与预热流程，不把先前探针历史混入一侧。
此配置的实际缓存命中路径仍需在推理设备验证，遇到兼容性错误应记录并修复，不以关闭缓存绕过。

代码支持范围：配置层会把enable_prefix_caching传入AR阶段，通用GPU runner会建立
OmniTensorPrefixCache。Hunyuan没有关闭完整hidden-state缓存的模型hook，因此除设备KV复用外，
还会维护CPU hidden-state缓存并在执行过程中拷贝数据；可能增加主机内存、锁页内存和同步开销。
当前基础配置仍关闭该开关，未找到Hunyuan PD缓存命中专项验证。本地没有安装vLLM和推理设备，
这里确认的是代码接入路径，不是P/D缓存命中与Mooncake传输组合已经运行验证。
PD侧仍设置do_remote_prefill并检查完整prompt KV接收；不能推断开启D侧缓存后P→D传输必然消失。
AR→DiT仍通过请求block表抽取KV，不能把prefix缓存命中直接等同为传输量同比减少。

Hunyuan明确实现step execution，代码包含逐步状态、合批及拆分逻辑，仓库有专项单元测试。
其限制为每请求一个prompt、一张图；不支持自定义timesteps/sigmas、sequence parallel>1、
CFG parallel>1、图像编辑、DiT计算缓存后端及DiT KV跳步。多请求要求TORCH_SDPA。
这里禁止的DiT计算缓存（如TeaCache）不等于AR prefix caching，两个开关没有这层直接互斥。
本次固定尺寸、n=1、默认50步的t2i输入符合请求侧约束，并行和缓存配置仍需核对实际部署。
开启后从整请求执行切换到逐去噪步调度，可能提高吞吐和减少等待，但也增加在途状态、显存及调度开销。
相同尺寸不是充分合批条件：模型还按首步/后续步、CFG分支数、latent形状、图像token数、
AR KV复用长度和是否有AR KV分组；max_num_seqs=2不保证每一步都以batch=2执行。
开启后的数值路径和注意力后端可能改变，固定seed也不能保证与原串行路径逐像素一致。

建议固定AR/P/D的 `max_num_seqs=32`，DiT为2，并启用DiT的step execution及TORCH_SDPA；
这些是需要在推理设备上验证的候选值。当前基础YAML的max_num_seqs仍为1，不能误当作已经调整。
同时关闭AR逐token调试和KV指纹；max_num_seqs及调试配置由服务启动配置管理，脚本不覆盖。
在最高目标并发运行long和mixed检查。失败先定位错误，必要时统一降低服务容量，再重跑对照。
客户端C与服务max_num_seqs不同：请求可以排队，全部完成不代表无排队、无抢占或容量最优。

该工具仅验证HTTP完成和实际PNG尺寸，不实现AR-only驱动、显存检查、AR TTFT/TPOT或KV计时。
完整t2i仍需同时采集内部AR指标和AR→DiT传输、DiT排队/执行时间；这些依赖服务端埋点。
响应时间包含curl启动及落盘；client_wall包含JSON/base64验证开销，不能用它发布正式吞吐结论。

正式矩阵保持短、长、混合均覆盖AR-only和完整t2i。按每卡并发配对，非PD/PD分别为：
AR-only 2/4、4/8、8/16、16/32；完整t2i 4/6、8/12、16/24。
该配比以原方案卡数为前提，换设备布局时需按实际卡数重算。
三轮选第三轮是暖机后的单次成绩，不是三次独立重复统计；保留前两轮用于诊断。

## 失败表现与影响

| 现象 | 含义与处理 |
|---|---|
| 启动时报错、服务未就绪 | 不应运行探针；检查YAML、内存与后端配置 |
| 输入重复、数量不足、混合奇数C、输出目录已存在 | 本地预检失败，尚未发送请求；修正参数或输入 |
| curl 7 / 28 / 连接重置 | 无法连接 / 总超时 / 连接断开；不能单凭这些判断OOM |
| HTTP 4xx | 检查model、参数、接口；通常不是容量问题 |
| HTTP 5xx、服务日志OOM或worker退出 | 检查服务日志；定位AR还是DiT，必要时降低对应阶段max_num_seqs |
| grouped DiT仅支持TORCH_SDPA的错误 | 配置DiT注意力后端，不用盲目调整AR容量 |
| HTTP成功但数量、PNG格式、实际尺寸错误 | 校验失败，不满足同尺寸实验条件 |
| 全部通过但服务有抢占/排队 | 只证明该负载完成；仍需独立采集性能和设备指标 |

请求失败会记录在roundN.json及原始响应文件中（未收到任何响应时可能没有文件），等待本轮其他请求
结束后退出非零，不运行后续轮次、不自动重试，不生成selected_result.json。原始响应可能是不完整JSON。
客户端超时或退出不保证服务器立即取消任务；开始下一组前确认无残留请求。worker崩溃时恢复服务。
较小max_num_seqs允许更多请求排队，不要求等于客户端并发。DiT从2降到1须对两种部署采用同一对照设置。
