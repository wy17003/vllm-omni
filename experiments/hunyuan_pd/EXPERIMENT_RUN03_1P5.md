# run03 补充实验：PD/NPD 并发比 1.5

参考 [PR #2220 Performance](https://github.com/vllm-project/vllm-omni/pull/2220)：
PR 按整套服务 3/2 的卡数比例设置 PD 为 NPD 1.5 倍并发，同时承认该方法并非完全公平。
这里沿用这一资源口径，模型仍为 HunyuanImage3，不直接套用 PR 的性能改善幅度。

## 1. 只补跑两个 NPD 点，复用已有 PD

| 负载 | PD，复用 hy3_run03 | NPD，本次执行 | 总服务卡数 | 请求数/总卡 |
|---|---|---|---|---|
| single | 48 个请求，一次发出 | 32 个请求，一次发出 | 12 / 8 | 4 / 4 |
| waves | 24+24，间隔 5 秒 | 16+16，间隔 5 秒 | 12 / 8 | 每波 2 / 2 |

每点只执行一轮。新启动 NPD，使用与 run03 相同的独立预热，再执行 single32、waves16+16。
共增加 64 个正式请求和 1 个预热请求。第二波不等待第一波图像返回，最大在途请求数为 32。
结果写入新的 `hy3_run03_1p5`，保留原 `hy3_run03` 的 PD/NPD 结果。

原 run03 的 48/24 是按 AR 卡数 8/4 归一；本次 48/32 是按总卡数 12/8 归一。
本次请求数/AR卡并不相等：single 为 PD 6、NPD 8。报告须明确这两个口径，不能混用。

复用 PD 的条件：服务源码、模型、设备/拓扑、配置、请求参数、观测方式保持与 run03 一致。
已提供的 PD 基线 Git HEAD 为 `0ff6fc182fd4e6b923699b600f448a1b7303b937`。
本次改动仅为客户端清单加载、选样及结果元数据；没有修改模型执行或服务观测代码。
若另行修改 P、KV 传输或服务配置，应另建实验对照，不能继续把旧 PD 当作同配置基线。

## 2. 输入已经准备好

使用 `schedule_run03_1p5/manifest.json`，无需额外 prompt 或 tokenizer 校准。

- PD 的两份 JSONL 与原 run03 **逐字节一致**，manifest 标记为 `reuse_run03`。
- 每个 NPD 点从对应 48 条 PD 输入中选 32 条，包含原 NPD 的全部 24 条，再增加 8 条。
- single：中文 24、英文 8；waves 每波：中文 12、英文 4。
- 按语言及波次匹配完整输入 token 均值；选样不使用输出质量、输出长度或性能成绩。
- single 与 waves 不重复，每批内不重复；PD/NPD 复用对应文本。
- 完整输入均值：single PD/NPD 为 1929.979/1929.969，waves 为 1928.333/1928.438。

构造程序为 `prepare_run03_1p5.py`，仅使用原冻结清单，无需模型权重。
新目录带 Git JSON/JSONL 例外及 LF 属性，避免漏传清单和 Windows/Linux 换行差异。
如确有必要在 Linux 重建，必须先保证原 `schedule_run03` 完整且通过哈希校验：

```bash
cd /home/w00934428/projects/vllm-omni || exit 1
python experiments/hunyuan_pd/prepare_run03_1p5.py \
  --source-dir experiments/hunyuan_pd/schedule_run03 \
  --output-dir /home/w00934428/hy3_run03_1p5_schedule_linux
```

默认直接使用仓库中已生成的 `schedule_run03_1p5`。重建时，执行命令的 `--schedule-dir` 改为新目录。
不手工修改 manifest 哈希或拿旧 NPD C24 文件冒充 C32。

## 3. 服务配置与执行指令

继续使用 `configs/run03_npd.yaml`：AR `max_num_seqs=64`、`max_num_batched_tokens=8192`，
prefix caching 和 chunked prefill 开启；DiT `max_num_seqs=1`、`step_execution=false`。
参数仍为 1024×1024、50 步、n=1、seed=42、guidance_scale=0、think/en_unified。
token 审计仍采用 `warn_equal_length_v1`。不加探针，不改变 5 秒间隔，不增加重复轮次。

同步本次客户端脚本及新清单；确认服务加载与 run03 一致的源码。
停止旧服务后**重新启动 NPD**，避免原正式输入已全部热缓存。不要在已有 NPD 会话上直接继续发请求。
保留原来正常使用的 NPU 环境，同主机运行客户端，服务无其他流量，日志不轮转。

**终端 A：启动 NPD，等待 ready。**

```bash
cd /home/w00934428/projects/vllm-omni || exit 1
mkdir -p /home/w00934428/hy3_run03_1p5
VLLM_OMNI_HY3_AR_TRACE=1 vllm serve /new_data/HunyuanImage-3.0-Instruct/ --omni \
  --port 9000 \
  --stage-configs-path /home/w00934428/projects/vllm-omni/experiments/hunyuan_pd/configs/run03_npd.yaml \
  --log-stats > /home/w00934428/hy3_run03_1p5/npd-server.log 2>&1
```

**终端 B：执行 NPD 两个点。**

```bash
cd /home/w00934428/projects/vllm-omni || exit 1
python experiments/hunyuan_pd/run_ar_experiment.py \
  --deployment npd \
  --schedule-dir experiments/hunyuan_pd/schedule_run03_1p5 \
  --tokenizer-json /new_data/HunyuanImage-3.0-Instruct/tokenizer.json \
  --output-root /home/w00934428/hy3_run03_1p5 \
  --server-log /home/w00934428/hy3_run03_1p5/npd-server.log \
  --timeout 3600
```

脚本按新 manifest 自动读取 NPD32，不需要额外 concurrency 参数。
本补充清单拒绝实际执行 `--deployment pd`，防止误补跑已完成的 PD；原 run03 清单仍支持原四点。
如使用 `--prepare-only`，另用全新 dryrun 输出目录，不能与正式输出共用。
失败保留全部数据并停止，不自动减并发、换 prompt 或重试。

## 4. 指标与结果配对

TTFT/TPOT、AR 完成时间、缓存命中、两波 token 间隔、E2E 均沿用 run03。
AR 窗口仍为：最后 AR 完成时间减第一次客户端发起时间，包含两波中间的 5 秒。

```text
AR tokens/s = 本批输出 tokens 总数 / AR 窗口秒数
tokens/s/AR卡 = AR tokens/s / (PD:8 或 NPD:4)
tokens/s/总卡 = AR tokens/s / (PD:12 或 NPD:8)
```

新增 `ar_trace.output_tokens_per_s_per_total_card`，保留原每 AR 卡字段。
`plan.json` 记录 `comparison_profile=total_cards_1p5x` 及两种卡数，避免混淆。
每总卡 AR 吞吐衡量为 AR 指标分摊的整套服务资源，不等于 images/s/卡；E2E 仍受串行 DiT 影响。

| 配对 | 既有 PD 结果 | 新 NPD 结果 |
|---|---|---|
| single | hy3_run03/pd/pd-single-n48…/selected_result.json | hy3_run03_1p5/npd/npd-single-n32…/selected_result.json |
| waves | hy3_run03/pd/pd-waves-n48…/selected_result.json | hy3_run03_1p5/npd/npd-waves-n32…/selected_result.json |

旧 PD 结果没有每总卡字段，可直接由已有 `output_tokens_per_s` 除以 12：
single 为 **20.7412**，waves 为 **21.5945 tokens/s/总卡**，无需改写原始结果。
同时保留原 NPD24 作为另一种资源口径的参照，不混合不同并发点计算均值。
观察两波影响时，沿用 run03 分析方法：对齐第二波实际开始 prefill，固定当时已在 decode 的第一波请求。

这次更改负载比例本身不修复已发现的 P 阶段长耗时，也不保证 PD 会胜出。
每点只有一次执行，不据此报告统计稳定性或置信区间。
结束后提供 `hy3_run03_1p5/npd` 和完整 `npd-server.log`；原 PD 材料继续使用此前提供的 run03。
