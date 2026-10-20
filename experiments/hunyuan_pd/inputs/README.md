# HunyuanImage3 PD 实验输入

基于测试分支 `test/support_hunyuanimage3_pd_disaggregation` / `ec5d4de05` 整理。
来源 `D:\Data\pdd\Exp` 保持原样；本目录包含加工后的输入、来源记录和重建工具。

## 数据与数量

| 文件 | 数量 | 用途 |
|---|---:|---|
| short.jsonl | 32 | 每主题一条短版 |
| long.jsonl | 32 | 相同32个主题的详细版 |
| mixed.jsonl | 64 | 同主题短长交替；任何偶数长度前缀均为1:1 |
| catalog.jsonl | 64 | 完整文本、配对ID、语言、来源、编辑说明及哈希 |
| schedule_manifest.json | — | 输入文件每行对应的主题ID和seed |
| prompts_preview.md | 32对 | 人工审阅长短版本 |
| audit.json | — | 来源数量、重复源、入选编号和字符统计 |
| calibrate_tokens.py | — | 在推理环境计算模型token长度 |
| build_inputs.py | — | 从本地源文件重建数据 |

此次在原16个主题基础上增加16个主题，生成32对、64条不同文本。
短版每条96–369字符，长版489–2402字符，每对长版至少是短版字符数的3倍。
每组25条中文、7条英文，同主题语言一致。短版提炼主体和风格；长版保留或补充具体场景、
材质、光照、布局细节，不用重复句子填充。长短改写可能改变AR思考内容和输出长度，
它是语义配对负载，不是只改变输入token数的严格消融。

`prompt.txt` 按空白段落划分为TXT-001至TXT-057；文档8个Show prompt块为MD-01至MD-08。
共65个候选块，TXT-029与TXT-030规范化后重复，只选029。保留原16个主题，新增
TXT-025/027/028/029/031/033/034/036/038/043/044/046/047/050/054/057。
TXT-047同段包含两个变体，只采用第二个；去除参考图依赖、填充队名和课堂知识点等占位符。
较短源文增加同主题场景细节，记录在catalog中。`prompt_utils.py`是改写提示词及代码，
不计为独立样本，也未执行。未选块不计入本次容量，并不意味着它们全部不可用。

**每批内禁止重复文本**，不通过更换seed扩充样本；三轮、PD/非PD复用对应输入（已确认）。
统一seed=42。旧的short_128、long_128和mixed_256重复负载已移除。
32条/纯组足以覆盖当前AR最大C=32及完整t2i最大C=24；混合只需分别16对和12对。
若未来纯组C>32或混合C>64，须补足独立文本，脚本不会循环取样。
不同并发取同一池的前缀，共享样本一致；不同请求数时不能保证整批相同或token分布完全匹配。

## 长度校准

以上是Unicode字符数，不是token数。公共system prompt会缩小完整输入的相对长度差，
中英文字符数也不可直接对比。目前未加载本地模型tokenizer，不把这些输入称作2K/4K/8K桶。
“详细版”是否足够长以触发PD收益仍需校准，不能仅凭文本数量作结论。

在推理服务器仓库根目录执行，仅加载tokenizer：

```bash
python experiments/hunyuan_pd/inputs/calibrate_tokens.py \
  --model /new_data/HunyuanImage-3.0-Instruct/
```

输出token_calibration.json，包括裸用户输入和通过本分支
`build_prompt_tokens(task='t2i', bot_task='think', sys_type='en_unified')` 构建的完整输入长度，
以及2K/4K/8K覆盖和组间重叠；同时核对服务端实际prompt_token_count。

## 尺寸与参数

全部输入固定width=1024、height=1024、size=1024x1024、n=1；在线探测去除width/height字段，
发送images API支持的size。竖幅9:16、3:4、2:3改为方形并重排构图，删除4K/8K画质口号和
其他工具命令。保留镜头焦距、光圈及画内竖条/四宫格等内容，它们不指定输出尺寸或图片数量。

显式size会影响think模式AR停止语义（`</think>` / `</recaption>`，而auto路径通常继续到比例token）。
给定curl未设置size，本实验保留显式size以保证同一实际尺寸；不能混用两条路径的成绩。
AR-only也必须复现模板、尺寸与停止语义。统一en_unified、think、50步、guidance_scale=0.0、seed=42。
最新要求禁止关闭prefix caching：基础部署目前仍为false，启动实验前需在所用配置中设为true，
并核对最终AR/P/D配置；客户端无法替服务切换。三轮复用输入可能命中缓存，第三轮是缓存开启、
重复输入预热后的成绩，需记录命中情况，不能全部归因于PD分离。探针默认一轮，正式实验仍为三轮取第三轮。

实际图片仍须在推理设备上验证PNG为1024×1024，不允许事后resize冒充相同工作量。
相同尺寸不保证相同AR KV复用长度或实际DiT合批率。当前未执行真实模型出图。

## 使用

参考上一级PROBE.md。探测脚本按group和C生成恰好N=C的batch.jsonl，支持prepare-only。
当前CustomDataset按num-prompts截取；若超过文件行数，则循环重复取样。正式benchmark应使用
生成的C行batch.jsonl，显式设num-prompts=C、max-concurrency=C、request-rate=inf，避免默认请求数
触发重复。batch.jsonl额外保留width/height，因为benchmark后端用它们构造size。
`catalog.jsonl`含元数据，仅用于审阅，不应作为HTTP压测输入。
完整t2i客户端接口不能提供全部内部AR/KV分解指标，仍需要服务端埋点；探测脚本不是完整性能采集器。
