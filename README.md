# Graduate_Paper_ghnn

基于结构保持神经网络的哈密顿系统研究 —— 毕业论文项目

## 项目简介

本项目是一个**毕业论文研究项目**，旨在实现和比较多种**结构保持神经网络（Structure-Preserving Neural Networks）** 在哈密顿动力学系统中的表现。项目核心围绕 **GHNN（Generalized Hamiltonian Neural Network，广义哈密顿神经网络）** 展开，通过与 MLP、SympNet、HenonNet 等基线模型的系统性对比，验证结构保持网络在能量守恒和长期轨迹预测方面的优势。

## 致谢与项目来源

> **重要说明：** 本项目的灵感与基础来源于 [AELITTEN/NeuralNets_GHNN](https://github.com/AELITTEN/NeuralNets_GHNN)。本仓库的前身即是 Fork 自该仓库。虽然本项目并未完全沿用其原始代码，但在模型设计思路、网络架构和实验方法上对其进行了模仿和借鉴，并在此基础上进行了扩展和改进。特此对原作者的工作表示感谢。

原始仓库实现了用于哈密顿系统的结构保持神经网络训练框架，本项目在此基础上：
- 重新编写了数据生成、训练和分析脚本
- 增加了多种基线模型（MLP、SympNet、HenonNet）的对比实验
- 在三个经典物理系统上进行了系统性的实验评估
- 对 GHNN 模型进行了针对性的改进和调试（如非可分系统的堆叠模块设计）

## 研究背景

哈密顿系统是经典力学中一类重要的动力学系统，具有能量守恒和辛结构（Symplectic Structure）等物理特性。传统神经网络（如 MLP）在学习此类系统时往往无法保持这些物理结构，导致长期预测中出现能量漂移和轨迹发散等问题。

**结构保持神经网络**通过在网络架构中嵌入辛积分器或哈密顿分解，能够在学习过程中自动保持物理守恒律，从而实现更准确、更稳定的长期预测。

## 模型架构

本项目实现并比较了以下四种神经网络模型：

### 1. MLP（多层感知机）— 基线模型
- 标准前馈神经网络，不包含任何结构保持机制
- 4 层隐藏层，128 维隐藏单元
- 作为对比基准，用于衡量结构保持方法的提升效果

### 2. SympNet（辛神经网络）
- 基于辛欧拉（Symplectic Euler）积分的神经网络层
- 实现正则辛更新：`p = p - h·F₁(q)`，`q = q + h·F₂(p)`
- 在积分过程中保持辛结构

### 3. HenonNet（Hénon 映射神经网络）
- 基于 Hénon 映射的辛积分器
- 实现：`p_new = p - h·f(q)`，`q_new = q + h·f(p_new)`
- 与 SympNet 类似的辛保持特性，但采用不同的数学构造

### 4. GHNN（广义哈密顿神经网络）— 核心模型
- 将哈密顿量分解为可分形式：`H(q,p) = U(q) + T(p)`
- 通过自动微分计算哈密顿梯度，实现辛欧拉积分：
  ```
  p_next = p - h · ∇_q U(q)
  q_next = q + h · ∇_p T(p_next)
  ```
- 对于非可分系统（如双摆、Hénon-Heiles），采用**多模块堆叠策略**（5 个独立可分模块的复合），近似一般哈密顿动力学
- 利用 PyTorch 的 `torch.autograd.grad` 实现可微分辛积分

## 实验系统

项目在以下三个经典哈密顿系统上进行实验：

| 系统 | 维度 | 哈密顿量类型 | 特点 |
|------|------|-------------|------|
| **单摆（Pendulum）** | 2D (q, p) | 可分 | 简单周期系统，基础验证 |
| **双摆（Double Pendulum）** | 4D (q₁, q₂, p₁, p₂) | 非可分 | 混沌系统，测试非可分哈密顿近似能力 |
| **Hénon-Heiles 系统** | 4D (x, y, pₓ, pᵧ) | 非可分 | 经典混沌系统，极端分布外泛化测试 |

### 数据划分策略
- **单摆**：按四分之一周期划分训练/测试集
- **双摆**：时间截断划分（训练 t∈[0,5]，测试 t∈[5,10]）
- **Hénon-Heiles**：能量壳约束 (E=0.125)，极端分布外测试（训练 t∈[0,10]，测试 t∈[10,50]）

## 项目结构

```
Graduate_Paper_ghnn/
│
├── 00_DATA_GENERATE/                  # 数据生成脚本
│   ├── my_generate_pendulum.py        #   单摆数据生成（Störmer-Verlet 积分器）
│   ├── my_generate_double_pendulum.py #   双摆数据生成（RK4 积分器）
│   └── my_generate_henonheiles.py     #   Hénon-Heiles 数据生成（Störmer-Verlet 积分器）
│
├── 01_MLP/                            # MLP 基线模型
│   ├── 01_pendulum/
│   │   ├── my_train_pendulum_mlp.py       # 训练脚本
│   │   └── my_analyze_pendulum_mlp.py     # 分析与可视化脚本
│   ├── 02_double_pendulum/
│   └── 03_henonheiles/
│
├── 02_SympNet/                        # SympNet 辛神经网络
│   ├── 01_pendulum/
│   ├── 02_double_pendulum/
│   └── 03_henonheiles/
│
├── 03_HenonNet/                       # HenonNet Hénon 映射网络
│   ├── 01_pendulum/
│   ├── 02_double_pendulum/
│   └── 03_henonheiles/
│
├── 04_GHNN/                           # GHNN 广义哈密顿神经网络（核心）
│   ├── 01_pendulum/
│   │   ├── ghnn_model_pendulum.py         # GHNN 模型定义
│   │   ├── my_train_pendulum_ghnn.py      # 训练脚本
│   │   └── my_analyze_pendulum_ghnn.py    # 分析与可视化脚本
│   ├── 02_double_pendulum/
│   │   ├── ghnn_model_double_pendulum.py  # 堆叠式 GHNN 模型定义
│   │   ├── my_train_doublependulum_ghnn.py
│   │   └── my_analyze_doublependulum_ghnn.py
│   └── 03_henonheiles/
│       ├── ghnn_model_henonheiles.py      # Hénon-Heiles 专用 GHNN 模型
│       ├── my_train_henonheiles_ghnn.py
│       └── my_analyze_henonheiles_ghnn.py
│
├── Results/                           # 实验结果（图表与可视化）
│   ├── Pendulum_GHNN/
│   ├── Pendulum_MLP/
│   ├── Pendulum_SYMPNET/
│   ├── Pendulum_HENONNET/
│   ├── DoublePendulum_GHNN/
│   ├── DoublePendulum_MLP/
│   ├── DoublePendulum_SYMPNET/
│   ├── DoublePendulum_HENONNET/
│   ├── HenonHeiles_GHNN_OOD/
│   ├── HenonHeiles_MLP_OOD/
│   ├── HenonHeiles_SYMPNET_OOD/
│   └── HenonHeiles_HENONNET_OOD/
│
├── Results_BACK_00/                   # 结果备份
├── .gitignore
└── README.md
```

## 环境依赖

### 必需依赖

| 依赖 | 用途 |
|------|------|
| Python 3.8+ | 编程语言 |
| [PyTorch](https://pytorch.org/) | 深度学习框架（含自动微分） |
| NumPy | 数值计算 |
| Pandas | 数据处理与 HDF5 读写 |
| Matplotlib | 数据可视化 |
| tables (PyTables) | HDF5 文件支持 |

### 安装依赖

```bash
pip install torch numpy pandas matplotlib tables
```

### 硬件建议
- **推荐使用 GPU**：所有训练脚本自动检测并使用 CUDA（如可用）
- 部分脚本含 GPU 显存管理（如 `torch.cuda.set_per_process_memory_fraction(0.22)`）

## 使用方法

### 第一步：生成数据

在项目根目录下运行数据生成脚本，生成的数据文件将保存到各子目录的 `Data/` 文件夹中：

```bash
# 生成单摆数据
cd 00_DATA_GENERATE
python my_generate_pendulum.py

# 生成双摆数据
python my_generate_double_pendulum.py

# 生成 Hénon-Heiles 数据
python my_generate_henonheiles.py
```

生成的数据格式为 HDF5（`.h5`），包含 `*_train.h5`、`*_test.h5` 和 `*_full.h5` 三类文件。

### 第二步：训练模型

进入对应的模型和系统目录，运行训练脚本：

```bash
# 示例：训练 GHNN 模型在单摆系统上
cd 04_GHNN/01_pendulum
python my_train_pendulum_ghnn.py

# 示例：训练 MLP 基线模型在双摆系统上
cd 01_MLP/02_double_pendulum
python my_train_doublependulum_mlp.py
```

训练产物（模型权重 `.pt`、预测结果 `.h5`、损失记录 `.txt`）保存在各子目录的 `NeuralNets/` 文件夹中。

### 第三步：分析与可视化

训练完成后，运行分析脚本生成结果图表：

```bash
# 示例：分析 GHNN 在单摆系统上的结果
cd 04_GHNN/01_pendulum
python my_analyze_pendulum_ghnn.py
```

分析脚本将生成以下内容：
- **相空间轨迹图**：真实轨迹 vs. 预测轨迹的对比
- **能量守恒图**：各轨迹的能量随时间变化曲线
- **MAE 随时间变化图**：平均绝对误差随预测步数的变化

### 主要训练超参数

| 参数 | 值 |
|------|-----|
| 隐藏层维度 | 60（GHNN）/ 128（MLP） |
| 隐藏层数 | 2–4 |
| 学习率 | 1e-3（Adam 优化器） |
| 批大小 | 256–512 |
| 最大训练轮数 | 2000–3000 |
| 时间步长 | 0.01 |
| 随机种子 | 2026 |

## 评估指标

本项目使用以下指标评估各模型的表现：

1. **能量守恒误差**：预测轨迹的哈密顿量相对于真实哈密顿量的偏差
2. **轨迹预测误差（MAE）**：预测状态与真实状态的平均绝对误差随时间的变化
3. **相空间结构保持**：在相空间中比较真实轨迹与预测轨迹的一致性
4. **分布外泛化能力**：在训练时间范围外的长期预测准确度

## 数据说明

由于数据文件（`.h5`）体积较大，未包含在仓库中。请使用 `00_DATA_GENERATE/` 中的脚本自行生成。训练产生的模型权重和预测结果同样通过 `.gitignore` 排除。

## 许可证

本项目仅用于学术研究和毕业论文目的。
