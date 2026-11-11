# GRF 信号特征提取与分类实验

本目录整理自本地 GRF.docx、Journal 文件夹和 OneDrive 源码压缩包。统一入口为 `main.py`，保留原始数据文件不作修改。此版本是可运行的整理与移植版本，不是原程序的逐字备份，也没有重新确认文档中的准确率。代码作者归属应以原项目记录为准，文档与代码注释不足以独立确认每段代码的作者。

## 环境与快速验证

使用 Python 3.12，在本目录执行：

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python verify.py
```

`verify.py` 自行生成临时模拟数据，验证两种输入表示、六种模型、身份分类和条件扩散流程。通过后清理临时数据。模拟数据指标不具有研究意义。CPU 可运行，不需要 GPU。依赖版本来自本次实际运行环境。

## 在现有真实数据上运行

```powershell
python main.py train --csv "F:/文章数据/Juefei_Yuan_journal/Journal/merged_features_6x18_with_gender_and_name.csv" --model nn --epochs 200 --output results/nn
python main.py train --csv "F:/文章数据/Juefei_Yuan_journal/Journal/merged_features_6x18_with_gender_and_name.csv" --model cnn --epochs 200 --output results/cnn
python main.py train --csv "F:/文章数据/Juefei_Yuan_journal/Journal/merged_features_6x18_with_gender_and_name.csv" --model rf --output results/rf
```

`--model` 支持 `rf`、`nn`、`cnn`、`svm`、`mlp`、`linear`。linear 是原 LinearCode 对应的回归基线，输出 MSE，不将回归数值冒充分类标签。输出目录包含预处理器、模型权重、指标和预测；神经网络另有训练日志。每个实验应使用独立输出目录。`--limit` 仅供快速验证，按文件前 N 行读取，可能只包含一个类别。

```powershell
python main.py extract --root "F:/文章数据/Juefei_Yuan_journal/Journal/gaitPatternsInCSVFiles/gaitPatternsInCSVFiles" --metadata "F:/文章数据/Juefei_Yuan_journal/Journal/gaitPatternsInCSVFiles/participantData.csv" --output results/features.csv
python main.py extract --root "F:/文章数据/Juefei_Yuan_journal/Journal/gaitPatternsInCSVFiles/gaitPatternsInCSVFiles" --metadata "F:/文章数据/Juefei_Yuan_journal/Journal/gaitPatternsInCSVFiles/participantData.csv" --raw --output results/raw.csv
python main.py train --csv results/raw.csv --raw --model cnn --output results/raw-cnn
python main.py train --csv results/raw.csv --raw --target index --model nn --output results/identity
python diffusion.py --csv "F:/文章数据/Juefei_Yuan_journal/Journal/merged_features_6x18_with_gender_and_name.csv" --epochs 200 --steps 100 --output results/diffusion
```

原始波形展开数据可能很大；先用 extract 的 `--limit 20` 检查路径和结构。元数据必须含 index、gender，按真实候选者 ID 匹配，不依赖目录枚举顺序。异常文件写入 `.errors.json` 并使操作失败，避免静默丢失数据。已有特征表有 13,702 行、108 个特征及 gender/candidate/source 三列。数据、论文、个人资料和大型压缩包未上传，使用者需自行提供数据。

## 原程序对应关系

| 原程序 | 整理版 |
|---|---|
| Journal/Training.py | main.py train，nn/cnn/rf 三种模式，为新增特征实验主程序 |
| Journal/read_and_save_data.py | main.py extract，统一 108 特征表 |
| Journal/to_juefei.py | to_juefei.py，仅保留 108 特征依赖的函数，移除无关导入 |
| Journal/SVM.py、SVM2.py | main.py train --model svm，统一固定 C=10 的 RBF 基线，未保留自动搜索 |
| Journal/MLP.py、MLP2.py | main.py train --model mlp，统一两层 128 单元模型，不等同于原两个架构 |
| Journal/transformer.py | diffusion.py，实际上是条件扩散增强加 MLP，并非 Transformer |
| 压缩包 ReadData_And_SaveToOneFile.py | main.py extract --raw |
| 压缩包 RandomForestCode_gender.py | main.py train --raw --model rf |
| 压缩包 NeuralNetworkCode_gender.py | main.py train --raw --model nn |
| 压缩包 ConvolutionalNeuralNetworkCode_gender.py | main.py train --raw --model cnn |
| 压缩包 NeuralNetworkCode_index.py | main.py train --raw --target index --model nn |
| 压缩包 LinearCode.py | main.py train --raw --model linear |

源文件散列见 sources.json。排除重复的 source code_old、IDE 缓存、历史结果以及与本研究无关的其他资料。原始文件全部保留在本地原位置。

## 方法与兼容性说明

新增 nn 保留 256/128/64/32 和 dropout 0.2；cnn 保留 64/128/64 卷积、批归一化和 0.3/0.3/0.2 dropout，均移植到 PyTorch，输出 logits 配合交叉熵。训练框架、随机初始化以及基线模型架构统一会影响结果，不保证复现 TensorFlow 原结果。

统一训练使用固定种子 42 的分层 50/25/25 试次划分，缺失特征填补和标准化仅在训练集拟合，最佳神经网络权重根据验证集选取。身份任务标签动态编码。条件扩散保留原实现，先按 80/20 划分，再从训练部分取 25% 验证，最终为 60/20/20；合成数据仅来自训练集，保存标准化参数，验证集选择权重，最终测试集只用于最终评价。

为兼容已有特征表，18 特征保留原名称和公式：13 个基础统计量、一个频域字典的数值均值、原 Sample_entropy、原 Higuchi、按采样点积分、按采样点拟合初始加载率。频域使用 100 Hz；积分与加载率没有转换为秒。原 Sample_entropy 仅比较相邻窗口，并非标准的全配对样本熵估计。这里保留用于历史兼容，不能当作标准熵实现。多种频域量混合成一列也应在后续方法研究中重新评估。NumPy trapz 改为 trapezoid 以兼容新版。

同一参与者可能出现在训练和测试两侧，因此试次级准确率不能代表对新参与者的泛化。文档中的 99.12% 等数值为历史记录，本次没有完成 200 epoch 的论文结果复现。validation.json 明确记录本次验证规模；核心模型使用完整现有特征表，神经网络仅 1 epoch；扩散验证用 256 条真实记录与 4 步采样。
