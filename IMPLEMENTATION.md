# 商品标题分类：实现与复现

本版本实现字符级 TF-IDF + LinearSVC，使用本地 `data/train.jsonl` 和 `data/test.jsonl`。原始数据文件保留。BERT、TextCNN 是选型讨论中的候选，本版本实现范围为已确认的线性分类方案。

本次开发运行的实验对比、指标、误差分析及性能限制见 [RESULTS.md](RESULTS.md)，最终验收仍以用户实际执行脚本的结果为准。

完整的分析因素、框架选择、流程优化、缺陷与后续设计见 [TECHNICAL_REPORT.md](方案.md)；实际实施过程见 [ITERATIONS.md](ITERATIONS.md)。BERT 微调与混合方案在报告中仅为设计，不具有运行入口或实验成绩。

## 运行

要求 Python 3.11 或更高版本、uv。仓库的 `.python-version` 选择 3.11。首次同步会安装锁文件中的依赖；有本地数据时，训练、测试及预测无需访问 ModelScope。

在项目根目录的 PowerShell 执行完整验收脚本：

```powershell
.\scripts\verify.ps1
```

只运行功能测试：

```powershell
.\scripts\verify.ps1 -UnitOnly
```

如果本机策略限制执行本地脚本，也可以逐条执行：

```powershell
uv --cache-dir .uv-cache sync --locked
uv --cache-dir .uv-cache run --no-sync pytest --basetemp .pytest-tmp
uv --cache-dir .uv-cache run --no-sync start audit
uv --cache-dir .uv-cache run --no-sync start train
uv --cache-dir .uv-cache run --no-sync start evaluate
uv --cache-dir .uv-cache run --no-sync start predict --input data/test.jsonl --output artifacts/predictions.jsonl --batch-size 4096
uv --cache-dir .uv-cache run --no-sync start benchmark --input data/train.jsonl --batch-size 4096
```

以上命令是可执行入口；效果和性能由实际运行生成，不预设达标阈值。验收脚本默认把结果写入 `artifacts/acceptance/`；独立命令默认模型目录为 `artifacts/tfidf/`。更换目录时，后续命令用 `--model-dir` 指向对应目录。

## 数据与划分

- 训练集 15,997 条、测试集 4,003 条，只有标题和十类标签。训练标题中位长度 14 字符，95% 不超过 21 字符。
- Unicode NFKC、大小写归一化和去除空白用于标题匹配及模型输入。数字、规格和其他字符保留。
- 用户明确决定：隔离归一化后同标题不同标签的记录，保存审计，不修改原始文件。当前为“令德堂冷榨芒果汁1.5L”的两条记录。
- 一致标签的重复记录保留，按归一化标题分组后，在训练文件内部按类别分层划出约 20% 验证集。同标题不跨训练、验证集合。词表和 IDF 仅在训练侧拟合。
- 官方测试集不参与模型选择。选出配置后，最终模型在隔离冲突后的完整训练集重新拟合。验证成绩属于选模阶段，测试成绩属于最终模型。
- 原始训练测试之间存在 85 个归一化重叠标题，涉及 86 条测试记录。评估保留官方测试集，同时分别报告训练中出现过和未出现过的标题成绩。它们是结果诊断，不用于调参。

## 实验记录

模型使用字符 2～4 gram、TF-IDF、sublinear TF、L2 归一化、最多 100,000 个特征、float32 稀疏矩阵。预先固定三组候选：`C=1`、`C=3`、`C=1 + class_weight=balanced`。随机种子默认 42，按验证集 Macro-F1 选择；并列时保留先出现的配置。

模型目录中保留：数据审计、三组实验指标及耗时、验证指标及错误样本、分组划分索引、最终模型、训练标题集合及元数据。元数据记录数据摘要、参数和依赖版本。需要复现实验时使用 `uv.lock`。

评估输出 Accuracy、Macro-F1、Weighted-F1、各类 Precision/Recall/F1/support、按固定十类顺序排列的混淆矩阵及错误样本。类别样本不均衡，因此主指标选 Macro-F1，另外关注冷藏果汁、口香糖等样本较少类别的召回率。

## 批量预测与性能

输入支持 UTF-8（可含 BOM）的 JSONL 和 CSV，必须有非空字符串字段 `product_name`。CSV 需要表头。其他输入字段不参与分类。输出为 JSONL 或 CSV，固定字段 `row_id`、`product_name`、`predicted_category`；`row_id` 为跳过空白行后的 1-based 记录序号，输出顺序与输入一致。

预测进程只加载一次模型，逐批读取、向量化、预测并写出。输出在处理完整个输入后替换目标文件；发生输入错误时保留已有目标文件。千万级输入无需一次性加载，内存还取决于模型词表、批大小及单条标题长度。

性能命令先加载模型并预热，再统计读取、解析、特征提取和分类的耗时，不包含结果文件序列化；预测命令另有包含写出过程的吞吐。RSS 为进程内存采样峰值（10ms 采样及批结束采样），可能漏过瞬时尖峰。报告记录硬件、批大小、进程线程数和运行环境。

`estimated_seconds_for_10_million` 是当前样本和机器条件下的外推，不能视为实际处理千万条的结果。完整验收脚本比较 256、1024、4096 三种批大小。

## 局限与后续建议

标题信息缺失和冲突标签会限制准确率；“冷榨”“NFC”不能直接作为冷藏标签的确定规则。字符特征仍可能利用品牌关联，在新品牌或不同来源数据上需要另行评估。当前没有独立的跨品牌测试划分，官方测试集重叠也可能让总体指标偏乐观。

后续若困难类别效果不足，可以在同一验证划分上比较中文 BERT；若 BERT 的收益明确，再测量额外推理成本。并发、量化、蒸馏等方向需要根据实测瓶颈另行决策。当前没有预先承诺准确率或千万级处理时限。

## 过程记录

1. 只读检查本地数据并比较 TF-IDF、BERT、TextCNN，基于短标题、局部品类信号及千万级预测成本，选择 TF-IDF + LinearSVC。
2. 用户确认方案，并选择隔离同标题异标签的两条记录，保留原始数据。
3. 实现本地数据审计、分组分层验证、三组候选实验、最终训练、独立评估和分批推理。
4. 可运行的功能测试检查数据隔离、持久化、批大小一致性、输出顺序及异常输入；真实数据结果由验收脚本生成。
