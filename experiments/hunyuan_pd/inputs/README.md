# HunyuanImage3 PD 实验输入

基于测试分支 `test/support_hunyuanimage3_pd_disaggregation` 整理。
当前完整方案见 [EXPERIMENT.md](../EXPERIMENT.md)，正式运行使用上一级schedule中的逐轮清单。
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

**每批内禁止重复文本**，不通过换seed扩充；跨轮、PD/非PD可以复用。
正式输入已在../schedule/冻结，包含PD24、非PD12/16各组各三轮，共27个文件。
每批25%英文，混合组严格长短各半；非PD为对应PD批次子集，每轮轮换部分主题。
按语言和同语言内字符长度近似匹配，无需模型tokenizer，无需人工再选输入。
目前纯组32条足够C24轮换，但不够三轮完全不重复；当前约束允许轮间复用。
文件SHA256、主题ID、字符分层统计和重合数见../schedule/manifest.json。

## token校准改为可选

字符数不是token数。公共system prompt会缩小完整输入相对长度差，中英文字符数不可直接当token比较。
短/长是现有文本分类，未宣称2K/4K/8K桶；本轮取消token校准前置要求。
如果后续需要解释长输入收益，可在推理环境运行calibrate_tokens.py，只加载tokenizer，
记录裸输入和build_prompt_tokens(task=t2i, bot_task=think, sys_type=en_unified)的完整长度。
不因性能结果反向筛选输入；本轮直接使用已经冻结的清单。

## 尺寸与参数

全部输入固定width=1024、height=1024、size=1024x1024、n=1；在线探测去除width/height字段，
发送images API支持的size。竖幅9:16、3:4、2:3改为方形并重排构图，删除4K/8K画质口号和
其他工具命令。保留镜头焦距、光圈及画内竖条/四宫格等内容，它们不指定输出尺寸或图片数量。

显式size会影响think模式AR停止语义（`</think>` / `</recaption>`，而auto路径通常继续到比例token）。
给定curl未设置size，本实验保留显式size以保证同一实际尺寸；不能混用两条路径的成绩。
所有请求统一en_unified、think、50步、guidance_scale=0.0、seed=42，始终为完整t2i。
prefix caching必须开启，本轮step_execution=false、DiT max_num_seqs=1，实验专用配置已提供。
轮换主题不清空缓存；不同部署和C的历史重复比例不同，第三轮不是同一批请求的第三次重复。
保留所有轮次，不把缓存开启下的全部收益归因于PD分离。

实际图片仍须在推理设备验证PNG为1024×1024，不允许事后resize冒充相同工作量。
本地尚未执行真实模型出图。

## 使用

推荐运行../run_experiment.py，按../EXPERIMENT.md先PD全部点，再非PD全部点。
客户端输出每轮roundN_inputs/batch.jsonl，恰好C行；若改用benchmark，需显式设置num-prompts=C，
否则CustomDataset可能循环取样。batch.jsonl保留width/height，因benchmark后端用它们构造size。
catalog.jsonl包含来源元数据，不是直接发送的HTTP请求正文。
