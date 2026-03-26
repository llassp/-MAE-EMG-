# MAE-EMG 

本项目实现了一个基于 Masked Autoencoders (MAE) 的肌电信号 (EMG) 处理与分析框架。项目包含了自监督预训练 (Pre-training) 以及下游分类任务 (Downstream Classification) 的完整代码，旨在探索自监督学习在时间序列/生物电信号上的应用。
并在kaggle公开数据集（https://www.kaggle.com/datasets/theseus200719/physiomio-16-gestures-64-channel-emg-dataset）上进行识别取得了97+%的正确率。

## 📁 目录结构

- **`model.py`**：模型定义文件，包含 MAE 编码器、解码器以及下游任务分类器的网络结构。
- **`dataset.py`**：数据处理脚本，负责加载 EMG 原始数据，并进行分片、归一化、掩码（Masking）等预处理操作。
- **`train.py`**：MAE 无监督预训练脚本，用于在大量未标注的 EMG 数据上重建信号，学习表征。
- **`downstream_train.py`**：下游分类任务训练脚本，基于预训练好的 MAE 编码器，进行监督学习微调。
- **`test_only.py`**：模型推理与测试脚本，用于评估验证集和测试集上的准确率或重建误差。
- **`visualize.py`**：可视化工具，可用于对比 EMG 信号的原始波形与 MAE 重建后的波形，直观展示模型的重建能力。
- **`requirements.txt`**：项目依赖清单。
- **`技术报告_EMG_MAE预训练.md`**：关于预训练过程、超参数设置、Mask 策略以及重建效果的详细技术文档。
- **`技术报告_EMG下游分类训练.md`**：关于下游动作分类/手势识别任务微调的技术验证、评估标准及结果报告。
- **`LICENSE`**：开源许可证文件。

## 🚀 环境依赖

请确保你的环境安装了 Python 3.8+，并使用以下命令安装必要的依赖库（例如 PyTorch, NumPy, Matplotlib 等）：

```bash
pip install -r requirements.txt
```

## 🧠 使用指南

### 1. 自监督预训练 (MAE Pre-training)
在没有标注的数据上训练 MAE 模型，使其学会重建被 Mask 掉的肌电信号片段。

```bash
python train.py
```
*提示：可根据需求在脚本内或通过 argparse 修改掩码率 (mask ratio)、学习率、批次大小 (batch size) 等参数。具体实验设置请参考 `技术报告_EMG_MAE预训练.md`。*

### 2. 重建效果可视化
训练完成后，你可以使用此脚本来查看输入信号、Masked 信号以及模型重建信号的对比：

```bash
python visualize.py
```

### 3. 下游分类训练 (Downstream Classification)
使用预训练的模型权重，在有标注的 EMG 数据��上训练分类器（如动作识别、疲劳检测等）。

```bash
python downstream_train.py
```
*有关下游任务的结构修改和性能评测结果，请参考 `技术报告_EMG下游分类训练.md`。*

### 4. 模型评估
在测试集上单独测试模型表现：

```bash
python test_only.py
```

## 📖 技术报告
我们在仓库中提供了两份详细的中文技术报告，包含了项目的理论基础、网络架构细节、实验对比分析等：
1. [MAE 预训练技术报告](./技术报告_EMG_MAE预训练.md)
2. [下游分类训练技术报告](./技术报告_EMG下游分类训练.md)

## 📄 许可证
本项目采用与 `LICENSE` 文件中声明的开源许可。详情请参阅项目中的 LICENSE 文件。
