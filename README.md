# All-program-for-my-paper

论文研究代码合集，集中整理 GRF 步态信号分析、病理显微图像碎片重建，以及 RAG/WAG 财务合规问答实验的程序。

三个子项目分别对应独立的研究流程，具有不同的数据格式、依赖和运行入口。本页提供整体导航与运行说明；算法细节和历史程序映射请结合子项目文档与源码阅读。

> 当前上传内容以源码、配置、提示词和少量演示数据为主。完整研究数据、论文原稿、预训练权重与完整实验输出未随本仓库提供。示例运行或流程检查通过，不等于已经复现论文中的性能指标。

## 项目导航

| 子项目 | 研究任务 | 主要方法与入口 | 详细说明 |
| --- | --- | --- | --- |
| [grf-analysis](grf-analysis/) | 地面反作用力（Ground Reaction Force，GRF）特征提取、性别分类与身份识别 | 108 维特征、原始六通道输入；RF、NN、CNN、SVM、MLP、线性回归基线；条件扩散增强 | [GRF README](grf-analysis/README.md) |
| [pathology-reconstruction](pathology-reconstruction/) | 病理显微图像碎片的邻接预测、位置与方向恢复 | 手工特征、传统机器学习、Feature MLP、PairNet、RF/PairNet 融合、GraphSAGE 与 GAT 重排序 | [配置](pathology-reconstruction/config.py) · [集成引擎](pathology-reconstruction/pathology_reconstruction_experiment.py) |
| [rag-wag-compliance-audit](rag-wag-compliance-audit/) | 内外部证据支持的财务合规问答与核验对比 | LLM Only、Static RAG、WAG Only、Hybrid RAG-WAG、Hybrid RAG-WAG + Runtime Verifier | [RAG/WAG README](rag-wag-compliance-audit/README.md) |

## 仓库结构

```text
All-program-for-my-paper/
├── README.md
├── grf-analysis/
│   ├── main.py                              # 特征提取与统一训练入口
│   ├── to_juefei.py                         # 每通道 18 项特征
│   ├── diffusion.py                        # 条件扩散增强与 MLP 分类
│   ├── verify.py                           # 自包含合成数据流程检查
│   ├── requirements.txt
│   ├── sources.json                        # 原始来源文件哈希
│   └── validation.json                     # 已记录的验证范围
├── pathology-reconstruction/
│   ├── config.py                           # 默认路径与实验配置
│   ├── data_pipeline.py                    # 数据准备、评估与输出
│   ├── pre_adapter.py                      # 加载集成实验引擎
│   ├── traditional_ml_suite.py
│   ├── feature_mlp_suite.py
│   ├── pairnet_suite.py
│   ├── fusion_graphsage_suite.py
│   ├── run_full_comparison.py
│   ├── pathology_reconstruction_experiment.py
│   └── pathology_reconstruction_*.py        # 五种模型的独立入口
└── rag-wag-compliance-audit/
    ├── src/                                # 检索、生成、核验与评估模块
    ├── scripts/                            # 实验、消融、检查与表格导出
    ├── prompts/                            # 提示词模板
    ├── examples/                           # 虚构问答与证据示例
    ├── requirements.txt
    └── source_manifest.json                # 已上传源文件清单与 SHA256
```

## 获取代码与运行环境

```bash
git clone --branch All-program-for-my-paper --single-branch https://github.com/yao920808-cmyk/All-program-for-my-paper.git
cd All-program-for-my-paper
```

当前分支名也是 `All-program-for-my-paper`。以下每个子项目的第一组命令均从仓库根目录开始执行，后续命令在该子项目目录中执行。

| 子项目 | Python 与依赖 | 硬件要求 |
| --- | --- | --- |
| GRF 分析 | 已有验证记录使用 Python 3.12；依赖版本见其 `requirements.txt` | 支持 CPU 运行 |
| 病理图像重建 | 建议使用 Python 3.12；NumPy、pandas、scikit-learn、scikit-image、OpenCV、PyTorch；XGBoost 可选 | 传统 ML、Feature MLP、独立 PairNet 对比可在 CPU 运行；集成重建入口要求 CUDA GPU |
| RAG/WAG 合规实验 | Python 3.10+，核心程序仅使用标准库；`.ps1` 包装脚本需要 PowerShell | Mock 演示无需 GPU 或 API Key；真实生成需要可用的模型服务配置 |

建议为不同子项目创建独立虚拟环境。下文 GRF 和病理项目的 Windows PowerShell 示例直接调用虚拟环境中的 Python，无需先激活环境；Linux/macOS 中对应路径为 `.venv/bin/python`。病理集成重建当前受 CUDA 要求限制。

## 一、GRF 步态信号分析

### 功能与输入

`main.py` 提供两条处理路径：从步态 CSV 提取每通道 18 项描述符，得到六通道共 **108 维特征**；或使用 `--raw` 将六通道原始采样值展开成表格后训练。

原始步态 CSV 需要包含以下列：

```text
leftVForce, leftAPForce, leftMLForce,
rightVForce, rightAPForce, rightMLForce
```

受试者元数据至少需要 `index` 和 `gender` 列。原始数据按受试者目录组织，例如 `data/trials/001/trial_01.csv`；提取程序按目录中的受试者 ID 匹配元数据。输出特征表含 108 个特征列及 `gender`、`candidate`、`source`；原始采样表则包含六通道及 `gender`、`index`、`source`。

### 安装与流程检查

```powershell
cd grf-analysis
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify.py
```

`verify.py` 自动生成临时合成数据，检查两种输入表示、六种模型、身份识别及条件扩散，并在结束后清理临时数据。该检查不需要原始研究数据，其输出不用于衡量论文性能。

### 使用自己的数据

以下 `data/` 路径是示例，需要准备对应文件或替换为自己的数据路径。

```powershell
# 提取 108 维特征
.\.venv\Scripts\python.exe main.py extract --root "data/trials" --metadata "data/participantData.csv" --output "results/features.csv"

# 随机森林分类
.\.venv\Scripts\python.exe main.py train --csv "results/features.csv" --model rf --output "results/rf"

# 神经网络分类
.\.venv\Scripts\python.exe main.py train --csv "results/features.csv" --model nn --epochs 200 --output "results/nn"

# 展开原始六通道采样值，并使用 CNN
.\.venv\Scripts\python.exe main.py extract --root "data/trials" --metadata "data/participantData.csv" --raw --output "results/raw.csv"
.\.venv\Scripts\python.exe main.py train --csv "results/raw.csv" --raw --model cnn --epochs 200 --output "results/raw-cnn"

# 原始采样表上的受试者身份识别
.\.venv\Scripts\python.exe main.py train --csv "results/raw.csv" --raw --target index --model nn --output "results/identity"

# 条件扩散数据增强，再训练 MLP 分类器
.\.venv\Scripts\python.exe diffusion.py --csv "results/features.csv" --epochs 200 --steps 100 --output "results/diffusion"
```

`--model` 接受 `rf`、`nn`、`cnn`、`svm`、`mlp`、`linear`。默认分类目标为 `gender`；`linear` 是回归基线，报告 MSE。提取可先加 `--limit 20` 检查数据结构，遇到无效文件时会生成 `.errors.json` 并终止，避免静默遗漏。

统一训练入口输出预处理对象、模型文件、`metrics.json` 和 `predictions.csv`；神经网络另有 `history.csv`。扩散入口保存扩散模型、分类器及预处理参数，最终分类评估打印到终端。每次实验应使用独立输出目录。

关于原始程序与整理后实现的对应关系、PyTorch 移植差异及特征公式，请阅读 [GRF 项目说明](grf-analysis/README.md)。

## 二、病理显微图像碎片重建

### 功能与程序入口

该项目先学习图像碎片边缘的邻接关系，再结合候选边评分、束搜索与模拟退火恢复碎片排列。输入可以来自完整图像切块，也可以使用现有切片目录；支持位置、方向及图像相似性评估。

| 程序 | 作用 |
| --- | --- |
| `traditional_ml_suite.py` | 比较 `color`、`texture`、`stain`、`fusion` 四类特征与 LR、SVM、RF、XGB 四种模型 |
| `feature_mlp_suite.py` | 比较四类手工特征上的 MLP 邻接分类器 |
| `pairnet_suite.py` | 比较 `rgb` 与 `rgb_hed` 两种输入的 PairNet |
| `fusion_graphsage_suite.py` | 读取传统 ML、Feature MLP 汇总，重训选中模型并比较重建结果，以及 PairNet、融合、GraphSAGE 方案 |
| `run_full_comparison.py` | 按默认配置依次运行传统 ML、Feature MLP、PairNet 与融合/GraphSAGE 对比 |
| `pathology_reconstruction_experiment.py` | 集成的训练、图重排序、搜索重建、批量评估和结果导出引擎 |
| `pathology_reconstruction_random_forest.py` | Random Forest 重建入口 |
| `pathology_reconstruction_pairnet.py` | PairNet 重建入口 |
| `pathology_reconstruction_fusion.py` | RF + PairNet 融合重建入口 |
| `pathology_reconstruction_graphsage.py` | 融合评分 + GraphSAGE 重排序入口 |
| `pathology_reconstruction_gat.py` | 融合评分 + GAT 重排序入口 |

### 环境与数据准备

此子项目尚无锁定版本的 `requirements.txt`。下面的依赖列表依据源码导入整理，适合搭建环境，不代表已固定的论文复现环境。

```powershell
cd pathology-reconstruction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn scikit-image opencv-python
```

随后在上述虚拟环境中安装与本机环境匹配的 **CUDA 版 PyTorch**，可参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。将官方命令中的 `pip` 换为 `.\.venv\Scripts\python.exe -m pip`，确保安装到本项目环境，再检查：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

集成重建需要最后一行输出 `True`。若要使用真正的 XGBoost，另行安装 `xgboost`；未安装时，当前代码会回退到 `HistGradientBoostingClassifier`，但输出标签仍为 `xgb`，分析结果时应记录实际实现。

将完整病理图像放入自备目录，例如 `data/images/`。程序直接读取该目录下的 PNG、JPG/JPEG、BMP、TIF/TIFF 文件。建议使用多个独立图像，并明确指定测试图像；只有一张图像时，代码会将其同时用于训练与测试。

若已有切片，可通过 `--slice_root` 指定切片根目录；切片目录命名与元数据解析规则见集成引擎中的 `find_slice_folder_for_image`、`read_slice_metadata` 等函数。没有现成切片时，使用 `--no_existing_slices` 从完整图像切块。

### 运行示例

```powershell
# 传统机器学习邻接分类对比
.\.venv\Scripts\python.exe traditional_ml_suite.py --train_dir "data/images" --test_image "data/images/test.png" --no_existing_slices --rows 4 --cols 5 --output_dir "outputs/traditional_ml"

# GraphSAGE 集成重建示例，训练轮数显式固定
.\.venv\Scripts\python.exe pathology_reconstruction_graphsage.py --train_dir "data/images" --test_image "data/images/test.png" --no_existing_slices --rows 4 --cols 5 --epochs 20 --max_epochs 20 --graph_epochs 8 --output_dir "outputs/graphsage"
```

数据路径和测试文件需自行替换。模型独立入口接受 `--rows`、`--cols`，也支持 `--grid_choice`：`1=3×6`、`2=4×5`、`3=5×4`、`4=5×7`、`5=10×8`、`6=5×5`。显式指定网格可避免交互选择。

模块化对比默认路径来自 [config.py](pathology-reconstruction/config.py)，当前仍包含现有本地 `F:` 盘路径。执行 `run_full_comparison.py` 前需先修改默认配置；该脚本当前使用默认解析参数，不转发命令行自定义路径。单独执行 `fusion_graphsage_suite.py` 时，还需要通过 `--traditional_summary` 和 `--feature_mlp_summary` 指向前序对比产生的汇总 CSV。

**硬件区别：** 独立的传统 ML、Feature MLP、PairNet 对比脚本可使用 CPU；调用 `run_experiment` 的五个模型入口、集成主程序及融合/GraphSAGE 对比中的集成阶段要求 CUDA GPU。集成入口的 `--cpu` 已禁用。

### 输出与评价

邻接分类阶段输出模型、各组合的 `metrics.json` 和 CSV/JSON 汇总，指标包括 Accuracy、Precision、Recall、F1、ROC-AUC 与 AP。

重建阶段输出 `original.png`、`scrambled.png`、`reconstructed.png`、`reference_reassembly.png`、`metrics.json`，并按模型保存权重和汇总表。重建指标包括位置准确率、邻接准确率、方向准确率、位置与方向联合准确率、SSIM、像素匹配比例和完全匹配状态。`reference_reassembly.png` 是参考拼接图，模型预测结果为 `reconstructed.png`。

## 三、RAG/WAG 财务合规问答实验

### 对比方案与处理流程

| 命令行方案名 | 对应方法 |
| --- | --- |
| `llm_only` | 不提供检索证据的 LLM 基线 |
| `static_rag` | 基于内部资料的检索增强生成 |
| `wag_only` | 基于外部/公开来源快照的检索增强生成 |
| `hybrid_rag_wag` | 结合内部与外部证据 |
| `hybrid_rag_wag_verifier` | 混合证据生成后增加运行时核验 |

流程包括问题分类、双通道词项检索、来源与日期检查、冲突分析、结构化回答、运行时核验，以及离线评分和错误分析。这里的 **WAG 使用预先准备的外部来源快照，不执行实时网页搜索**。

### 离线快速开始

以下 PowerShell 命令使用仓库自带的虚构示例，无需 API Key：

```powershell
cd rag-wag-compliance-audit
.\scripts\run_prepared_step6.ps1 -Mock -QaCases examples\example_qa_cases.csv -Chunks examples\example_chunks.csv -OutputStem example
python scripts/08_validate_smoke_outputs.py --model-outputs outputs/model_outputs/example_model_outputs.jsonl --output outputs/evaluation_results/example_validation.csv --expected-rows 10
```

示例包含 2 个问题，每个运行 5 种方案，共生成 10 条模型输出。结果保存在 `outputs/model_outputs/`、`outputs/verifier_outputs/` 和 `outputs/evaluation_results/`。Mock 用于检查流程与文件结构，不调用真实模型，也不代表研究性能。

Bash 用户可在该项目目录执行 `bash examples/run_example.sh`，运行生成、离线评分与指标汇总三个步骤。

### 自备研究数据与真实模型

问答 CSV 包含 `qa_id`、`question_type`、`question`、`expected_answer`、`expected_evidence_id`、`conflict_type`、`requires_human_review` 等字段；证据 CSV 包含 `chunk_id`、`source_type`、`title`、`chunk_text`。具体格式见 [examples](rag-wag-compliance-audit/examples/)。

完整的 470 条研究问答集、内外部语料、来源快照、交易数据和人工评审记录未包含在当前上传中。部分脚本默认读取相邻的 `prepared_data/experiments/`，使用自己的数据时应显式提供输入路径。

真实生成通过 `scripts/01_run_five_models.py` 或 PowerShell 包装脚本运行；需要先配置服务凭据，明确指定账号可用的模型 ID，并移除 Mock 选项。源码包含 OpenAI、OpenRouter 和 Gemini 适配，具体配置及安全输入密钥的示例见 [子项目 README](rag-wag-compliance-audit/README.md) 与 [llm_client.py](rag-wag-compliance-audit/src/llm_client.py)。真实请求可能产生服务费用。更改模型、提示词或数据时应使用新的输出名称；`--resume` 仅跳过已存在的问题/方案组合，不检查配置是否一致。

### 实验脚本

| 脚本 | 用途 |
| --- | --- |
| `01_run_five_models.py` | 运行五种对比方案，可选择其中部分方案 |
| `02_run_verifier.py` | 基于标准答案及标签的离线评分 |
| `03_evaluate_results.py` | 汇总自动评价指标 |
| `04_run_ablation_study.py` | RAG、WAG、核验、日期检查、来源排序等组件消融 |
| `05_error_analysis.py` | 导出错误案例 |
| `06_lock_experiment_inputs.py` | 准备数据划分、输入哈希及来源记录 |
| `07_build_publication_tables.py` | 生成论文表格所需的 CSV |
| `08_validate_smoke_outputs.py` | 检查输出结构、引用 ID 及真实响应元信息 |
| `09_make_balanced_smoke_subset.py` | 按问题类型抽取流程检查子集 |
| `10_apply_runtime_verifier.py` | 对已有模型输出重放运行时核验，无需重新请求模型 |

回答包含合规结论、支持证据、冲突分析、生效日期分析、人工复核标记和限制说明。自动指标包括 `answer_accuracy` 代理指标、`retrieved_citation_match`、证据 Precision/Recall、启发式幻觉率、冲突与日期处理、人工复核标记准确率等。

## 数据、验证与结果解释

| 项目 | 仓库已包含 | 完整研究仍需自行准备 |
| --- | --- | --- |
| GRF 分析 | 源码、固定依赖、原始来源哈希、历史验证记录、合成数据检查程序 | 原始步态 CSV、受试者元数据或已有特征表、完整训练结果 |
| 病理重建 | 重建引擎、模型入口、模块化对比与路径配置 | 原始图像、已有切片及其元数据（如使用）、匹配的运行环境与实验结果 |
| RAG/WAG | 源码、提示词、脚本、源文件清单、2 条虚构问答与 3 条虚构证据 | 完整问答和语料、可追溯外部快照、人工评审、真实模型输出 |

使用结果时请结合当前实现的评价边界：

- **GRF：** 统一训练入口使用固定种子 42 的分层行级划分，训练/验证/测试为 50%/25%/25%；扩散流程为 60%/20%/20%。同一受试者可能进入不同集合，结果不等同于对未见受试者的泛化性能。历史特征保留了非标准 `Sample_entropy` 等兼容实现，公式解释见子项目文档。
- **病理重建：** 优先按图像划分训练和验证，数据不足时会回退到碎片对层面划分。集成流程会依据参考重建指标从搜索候选及逐轮结果中选择最佳结果；这些指标不能直接视为严格盲测结果。正式对比需固定划分、训练轮数、搜索预算及选模规则，并使用独立测试数据。
- **RAG/WAG：** 当前检索基于词项重叠；来源排序虽被记录，真实生成提示词仍使用各通道检索列表，排序消融不能据此直接证明生成收益。引用匹配检查的是 ID 是否属于检索集合，不能证明证据在语义上支持结论。自动回答准确率和幻觉检查采用代理规则，需要结合人工语义评审。
- **核验区别：** RAG/WAG 的 Runtime Verifier 不使用标准答案标签；离线 scorer 使用预期标签。正式实验需明确可提供给运行时的元数据，防止标准答案或评估标签成为生成提示。

[GRF validation.json](grf-analysis/validation.json) 记录了既有验证范围，包括神经网络单轮训练和小规模扩散检查，不代表完成了论文中的完整训练。本次 README 整理实际运行了 RAG/WAG 的离线 Mock 流程，10 条输出通过结构检查；未运行完整 GRF 或病理训练，也未执行真实 LLM 实验。

复现实验时建议保存代码提交号、Python/依赖版本、随机种子、数据划分、输入哈希、完整参数及输出目录。真实 LLM 实验还应保存模型 ID、提示词版本和外部资料快照；温度为零不保证跨次响应完全一致。

## 来源、引用与许可

[grf-analysis/sources.json](grf-analysis/sources.json) 记录原始来源文件哈希；[rag-wag-compliance-audit/source_manifest.json](rag-wag-compliance-audit/source_manifest.json) 记录该子项目已上传源文件清单与 SHA256。引用时请注明对应论文与实际使用的代码提交版本，具体作者归属以原始项目记录为准。

当前仓库未附带统一的开源许可证。使用或再分发代码、数据及第三方材料前，请核对相应来源的授权条件。
