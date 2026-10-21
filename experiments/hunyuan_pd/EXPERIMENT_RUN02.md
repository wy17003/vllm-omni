# HunyuanImage3 PD 补充性能实验（run02）

本轮继续完整 t2i，采集 TTFT、TPOT、E2E。复用 run01 的 PD short/mixed 第三轮，
替换风险输入后补跑 PD long 三轮，再执行 NPD 全部六个点。
不修改推理代码、模型、采样参数或服务配置；本轮修改仅涉及客户端实验工具、离线分析及输入清单。
原始结果和 schedule 保留，所有新结果写入新目录。

2026-10-21 验收口径更新：采用 `warn_equal_length_v1`。解码文本一致且仅发生等长 token ID 差异时，
记录警告并继续；长度变化、文本不一致、日志缺失/冲突仍停止。PD/NPD及离线重算使用同一规则。
这仅调整性能实验的验收范围，不是修复KV对齐缺陷，也不是图像等价性证明。
旧 run02 round1 因严格审计停止的结果保留；更新后按第6节在 `hy3_run02_warn` 从三轮重新开始。

## 1. 数据来源与实验矩阵

| 输入组 | PD C24 | NPD C12 | NPD C16 |
|---|---|---|---|
| short | 复用 run01 第三轮 | 新清单执行三轮 | 新清单执行三轮 |
| long | 新清单补跑三轮 | 新清单执行三轮 | 新清单执行三轮 |
| mixed，短长各半 | 复用 run01 第三轮 | 新清单执行三轮 | 新清单执行三轮 |

- AR 子系统归一：PD P4+D4 对 NPD AR4，C24/C12，concurrency/card=3。
- 完整服务归一：PD 共12卡对 NPD 共8卡，C24/C16，concurrency/card=2。
- 所有点均执行完整 t2i；没有新增 AR-only 模式。AR 指标仍可能受完整管线调度影响。
- 每轮请求数等于并发数，三轮轮换既定输入，固定第三轮为主结果，不取最快一轮。
- 本轮新增 PD 72 请求、NPD 252 请求，共21批、324请求。复用 PD short/mixed 第三轮共48条主结果。
- 这是有限突发批次实验，不是持续饱和吞吐实验。

run01 的源代码 HEAD 为 ec7d8a281。请在同一测试分支同步本轮实验文件，不升级推理依赖或模型。
新客户端 Git HEAD/diff 与旧记录可以不同，但推理实现必须保持一致；驱动会保存 Git 记录。

## 2. 为什么复用这些结果

已核对 run01 PD 三组九轮的216条请求：全部 HTTP 成功且客户端记录尺寸1024×1024。
token 往返审计发现：

| 位置 | 输入 | AR 输出 tokens → 文本重编码 tokens | 处理 |
|---|---|---|---|
| PD long 第2轮 | scene19_long | 458 → 458，但 ID 不同 | 保留异常记录 |
| PD long 第3轮 | scene11_long | 525 → 524 | 旧 long 不进入本轮主对比 |
| PD mixed 第2轮 | scene11_long | 540 → 539 | 披露前序轮异常；复用无此差异的第3轮 |
| NPD short C12 第1轮 | scene31_short | 466 → 467，HTTP500 | 失败批次不作为正式成绩 |

short 三轮及 mixed 第三轮未发现这种差异。复用结论仅覆盖已核对的输入、日志与已知缺陷，
不等于完成数值 KV 或图像质量等价验证。旧 mixed 第二轮的异常不会从记录中删除。

原 summary.json 只包含 short，是旧汇总快照；离线工具从实际点目录重建完整汇总。
旧 long 的 AR 指标仍可供诊断，但不能删除一个异常请求后将23条作为 C24 主结果，
也不能改选旧第一轮代替第三轮。

## 3. 输入清单已冻结，无需再次选 prompt

使用 `experiments/hunyuan_pd/schedule_v2/manifest.json` 和对应27个 JSONL。
其中 PD short/mixed 六个 JSONL 是原输入的引用，action=reuse_run01，驱动不会重新执行。
其余21个 JSONL 用于本次运行。原始 `schedule/` 不修改。

新执行批次排除 scene31_short、scene11_long、scene19_long；历史 PD 引用仍保留原样。
只替换风险项，其他输入及位置不变。替换来自原64条输入池，不改写 prompt，不换 seed 扩充。
按相同语言、相同长短类别、最接近的完整 AR 输入 token 数选择，字符数和 ID 用于确定性排序。
完整输入 token 数取自 run01 现有日志，无需重新加载模型或在线校准；没有用耗时/加速比选样。

PD long 的替换如下：

| 轮次 | 原输入 | 新输入 | 完整输入 tokens |
|---|---|---|---|
| 1 | 无替换 | 保持原批次 | — |
| 2 | scene11_long | scene25_long | 1594 → 1587 |
| 2 | scene19_long | scene17_long | 2104 → 1731 |
| 3 | scene11_long | scene24_long | 1594 → 1585 |
| 3 | scene19_long | scene03_long | 2104 → 1850 |

较长样本的替代差距来自“批内不重复、保留其他输入”的约束，并不声称逐条等长。
NPD 每轮仍是对应 PD 批次子集；short/mixed 对应旧 PD，long 对应新 PD。
NPD 相对对应 PD 的整批平均完整输入 token 数偏差：short≤0.04%、long≤1.04%、mixed≤1.30%。
这不保证尾部分布或 AR 输出长度完全一致。全部替换及原/新长度见 manifest 的 replacements。
例如 NPD short C12 第1轮改为 scene21_short，完整输入同为1312 tokens，
并非之前只按字符数举例的 scene28_short。

每批24/12/16条互不重复，英文25%、中文75%，mixed严格短长各半；允许跨轮和部署复用。
现有输入足够，不需要补 prompt。轮换不是同一批输入的三次重复。
新 JSONL 固定 LF，目录内 .gitattributes 固定检出换行，避免原先 Windows/Linux 字节哈希差异。
不要手工修改 JSONL 后更新哈希来绕过校验。

优先复用服务器已经生成的 `/home/w00934428/hy3_run02/schedule_linux`，无需因规则调整重新选样。
仓库全局忽略 JSON/JSONL，提交1042fddd9仅包含schedule_v2/.gitattributes，不能假设git同步带来了完整清单。
若服务器尚无清单，使用以下命令生成一次（输出目录必须不存在）：

```bash
python experiments/hunyuan_pd/prepare_revision.py \
  --baseline-dir /home/w00934428/hy3_run01/pd \
  --tokenizer-json /new_data/HunyuanImage-3.0-Instruct/tokenizer.json \
  --output-dir /home/w00934428/hy3_run02/schedule_linux
```

## 4. 固定资源、服务与请求参数

| 设置 | PD | NPD |
|---|---|---|
| 阶段设备 | P:0–3；D:4–7；DiT:8–11 | AR:4–7；DiT:8–11 |
| TP | 每阶段4 | 每阶段4 |
| max_num_seqs | 32 / 32 / 1 | 32 / 1 |
| enable_prefix_caching | true | true |
| DiT step_execution | false | false |
| DiT attention | TORCH_SDPA | TORCH_SDPA |
| AR max_num_batched_tokens | 8192 | 8192 |
| AR / DiT memory utilization | 0.8 / 0.65 | 0.8 / 0.65 |

固定请求为1024×1024、n=1、50步、seed=42、bot_task=think、use_system_prompt=en_unified、
guidance_scale=0、response_format=b64_json；保持 run01 显式 guidance_scale 的源码修正。
配置来自 experiments/hunyuan_pd/configs/，不使用之前手工修改过的基础 deploy YAML。
启动前移除可能覆盖配置的旧调试环境变量；保留源码已有的 HY3_EQ AR 输入/输出日志，
不需要重新开启逐 token 调试或 KV fingerprint。

当前为最少补跑方案：新 PD long 从新服务开始，旧 PD short/mixed 保留原连续运行历史；
NPD 按固定顺序连续执行，不在点之间清空缓存。新 long 缺少旧 PD 的 short 前序流量，
因此不能将它描述为原 run01 的原位重测，也不能单独把 TTFT 差异归因于 PD。
这项缓存历史限制必须随结果披露；如果要求严格相同的冷/暖缓存历史，需要另做控制实验。
不调用当前未实际清理缓存的 reset_prefix_cache 接口。

## 5. 先离线恢复 run01 PD 指标

使用现有推理环境的 Python 和 tokenizers 包；分词审计只加载 tokenizer.json，不加载模型权重。
以下操作不需要运行服务，也不修改原始数据：

```bash
python experiments/hunyuan_pd/reparse_results.py \
  --deployment pd \
  --source-dir /home/w00934428/hy3_run01/pd \
  --output-dir /home/w00934428/hy3_run01_pd_analysis_warn \
  --tokenizer-json /new_data/HunyuanImage-3.0-Instruct/tokenizer.json
```

输出目录必须不存在。预期 third_round_eligible：short=true、mixed=true、long=false。
每轮结果包含重算的 ar_metrics、token_audit、analysis_eligible 和源文件哈希。
source_valid保留原执行判定；valid根据原HTTP/图像记录及当前审计规则重算，
analysis_eligible还要求TTFT/TPOT完整。离线重算不会修改源文件，也不会补齐未执行的轮次。
summary.json 从全部实际点目录重建，不依赖旧的不完整 summary。

解析规则：同请求同阶段的重复列（如1与1_2、0与0_2），仅在全部已打印字段一致时计一次。
任何冲突、缺失、非正值或请求覆盖不完整仍为 N/A，不相加、不用零替代缺失。

已核对的第三轮均值，供确认重算结果：

| 组 | TTFT ms | TPOT ms/token | E2E s | 本轮用途 |
|---|---:|---:|---:|---|
| short | 613.128 | 91.849 | 300.210 | 复用 |
| long | 1098.452 | 91.439 | 323.945 | 仅诊断 |
| mixed | 1031.669 | 90.912 | 329.664 | 复用，披露前序轮异常 |

## 6. 先补跑 PD long，再执行 NPD 全部点

在推理服务器仓库根目录运行。保持独占服务流量；日志应为客户端本地可读的追加文件，不能轮转。
停止此前失败/筛选用服务，确认进程退出后再开始。脚本不负责启动服务。
新规则输出写入hy3_run02_warn，与原run01、已停止的run02分开；禁止覆盖原目录。
驱动不支持从旧round1目录恢复round2；使用新目录完整执行三轮，原round1保留为诊断记录。
更新只涉及客户端工具；重新启动服务是为重新开始本轮缓存历史，不涉及模型/服务实现修改。

**终端A：启动 PD，等待服务 ready**

```bash
cd /home/w00934428/projects/vllm-omni || exit 1
mkdir -p /home/w00934428/hy3_run02_warn
vllm serve /new_data/HunyuanImage-3.0-Instruct/ --omni \
  --port 9000 \
  --stage-configs-path /home/w00934428/projects/vllm-omni/experiments/hunyuan_pd/configs/pd.yaml \
  --log-stats > /home/w00934428/hy3_run02_warn/pd-server.log 2>&1
```

**终端B：只执行 PD long 三轮**

```bash
python experiments/hunyuan_pd/run_experiment.py \
  --deployment pd --groups long \
  --schedule-dir /home/w00934428/hy3_run02/schedule_linux \
  --tokenizer-json /new_data/HunyuanImage-3.0-Instruct/tokenizer.json \
  --output-root /home/w00934428/hy3_run02_warn \
  --server-log /home/w00934428/hy3_run02_warn/pd-server.log \
  --timeout 1800
```

三轮均通过后停止 PD，确认进程、设备和端口释放，再启动 NPD。若任何轮次失败，不继续正式对比。

**终端A：启动 NPD，等待服务 ready**

```bash
vllm serve /new_data/HunyuanImage-3.0-Instruct/ --omni \
  --port 9000 \
  --stage-configs-path /home/w00934428/projects/vllm-omni/experiments/hunyuan_pd/configs/npd.yaml \
  --log-stats > /home/w00934428/hy3_run02_warn/npd-server.log 2>&1
```

**终端B：执行 NPD 六个点，每点三轮**

```bash
python experiments/hunyuan_pd/run_experiment.py \
  --deployment npd \
  --schedule-dir /home/w00934428/hy3_run02/schedule_linux \
  --tokenizer-json /new_data/HunyuanImage-3.0-Instruct/tokenizer.json \
  --output-root /home/w00934428/hy3_run02_warn \
  --server-log /home/w00934428/hy3_run02_warn/npd-server.log \
  --timeout 1800
```

顺序为C12 short→long→mixed，再C16 short→long→mixed。不额外做容量探针，不修改并发或max_num_seqs。
`--groups` 只用于筛选要运行的输入组；schedule_v2 的 PD short/mixed 标记为复用，驱动不会重跑。
运行目录仍记录 pc-on、step-off 和各阶段 max_num_seqs，但这些是客户端声明，不是服务器读回。
请保存完整启动日志用于核对实际配置。

可选在正式启动前加 `--prepare-only`，将 `--output-root` 改成全新的 dry-run 目录，
并去掉 `--server-log`。它只生成请求，不向服务发请求；仍校验清单和分词器文件。

## 7. 每轮验收与失败处理

run02 清单强制要求 --tokenizer-json，且文件SHA必须与冻结清单一致。
每批HTTP请求结束、计时停止后，用已有日志自动检查：

1. 该批实际输入能与 AR 输入日志唯一对应，完整覆盖24/12/16条请求。
2. 原始 AR 输出 token 解码文本与日志一致。
3. 相同生成文本重新编码后，token数量必须一致；同长度但ID不同只告警，不停止。

结果中的token_audit.policy为warn_equal_length_v1：

- warnings：等长ID差异，要求解码文本与日志一致；打印WARNING并允许后续轮次继续。
- blocking_mismatches：token数量增减或解码文本不一致，仍停止。
- errors：日志缺失、覆盖不完整、无法唯一映射或日志冲突，仍停止。
- mismatches保留全部差异；roundtrip_consistent=false明确表示token并非完全一致，即使valid=true。
- 每轮保留完整差异；selected_result及部署summary附带三轮token_audit_round_warnings，不能只报告无警告样本。

HTTP/尺寸校验或 token 审计任一失败，该轮 valid=false，保存错误和token差异后停止后续轮次，
不产生该点 selected_result。不自动重试、不自动换 prompt，不选择最快或偶然成功的一轮。
失败发生在第三轮时，前两轮也不能替代主结果。
审计缺少必要日志也会失败，避免把“没检查到”当作通过。
TTFT/TPOT若为N/A，检查conflicts及日志覆盖；不进入对应指标的正式对比。

审计在客户端计时结束后执行，不计入 E2E/client_wall；它会增加批次间隔，原 run01 没有这个检查，
这一运行节奏差异应记录。检查只针对已知文本重编码缺陷，不是完整KV/画质验证。

若出现阻断项，保留结果和新风险输入，停止并分析；等长警告按上述规则继续，不再据此替换prompt。
不要反复筛选直到获得更好的性能。
替换版本必须重新冻结并说明原因，不能覆盖已有输入和结果。

## 8. 汇总与交付

最终对比取以下来源，不把新旧目录混成一次连续运行：

| 数据 | 来源 |
|---|---|
| PD short/mixed | hy3_run01_pd_analysis_warn 对应第三轮 |
| PD long | hy3_run02_warn/pd 对应第三轮，要求审计无阻断项，保留警告 |
| NPD short/long/mixed，C12/C16 | hy3_run02_warn/npd 各点第三轮，同一审计规则 |

报告每项样本数、均值、P50和最大值：TTFT取PD stage0/NPD stage0，TPOT取PD stage1/NPD stage0。
PD TTFT不含完整P→D等待，不能与D的TTFT相加；E2E沿用客户端逐请求响应时间。
原始三轮全部保留，不把轮换输入当成同一负载重复测量来计算稳定性。
当前长输入只是相对分类，不称为2K/4K/8K桶；新PD long第三轮平均完整输入约1712 tokens。

提供 hy3_run02_warn/pd、hy3_run02_warn/npd 的 summary、plan、selected_result、roundN.json、
roundN_server.log、roundN_inputs/requests.json、manifest、配置和Git记录。
同时保存完整 pd-server.log/npd-server.log 的启动部分；暂不要求传输大体积图片响应。

本地验证覆盖已有216条PD日志重算、输入清单唯一性/比例/子集关系、token异常检测、
重复阶段冲突检测和真实curl调用模拟HTTP服务。实际NPU补跑必须在推理服务器执行。
