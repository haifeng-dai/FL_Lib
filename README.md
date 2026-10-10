# FL_MP

FL_MP 是一个基于 PyTorch 多进程的联邦学习实验框架，面向单机多 GPU 训练。项目实现了全局联邦学习、个性化联邦学习、去中心化联邦学习和组合式联邦学习算法，并提供数据划分、实验记录与结果比较工具。

## 功能概览

- 支持 CIFAR-10、CIFAR-100 和 MNIST，支持 IID、Dirichlet 和病态标签划分。
- 通过统一命令行入口运行算法；算法参数与通用训练参数可组合使用。
- 使用 PyTorch 多进程调度 GPU worker，可配置每张 GPU 的 worker 数量。
- 将参数、日志、逐轮指标、checkpoint 和运行摘要保存到结果目录及 SQLite 数据库。
- 提供多试次运行、批量实验、结果查询、算法比较和试次聚合命令。
- 提供按算法类别组织的 Jupyter Notebook 结果分析示例。

## 算法

算法通过 `--algo` 指定，名称对应 `algo/` 下的 Python 文件名。

| 类别 | 算法 |
| --- | --- |
| 全局联邦学习（GFL） | FedAvg、FedProx、FedDyn、MOON、SCAFFOLD、FedProc、FedFM、FedPLN、FedLSA |
| 个性化联邦学习（PFL） | Local、FedPer、FedRep、FedProto、FedTGP、FedSA、FedKD、FedDPC、FedALA、FML、LG-FedAvg |
| 去中心化联邦学习（DFL） | DisPFL、DFedPGP、DFedSet、DFedAvgM、L2C、ProxyFL、EFHC、PearFL |

具体参数因算法而异，可通过帮助信息查看：

```bash
uv run main.py --algo fedavg --help
```

## 环境与安装

- Python 3.12 或更高版本
- NVIDIA GPU、CUDA 驱动及与本机环境兼容的 PyTorch
- [uv](https://docs.astral.sh/uv/)

在项目根目录安装依赖：

```bash
uv sync
```

`pyproject.toml` 为 Linux 和 Windows 配置了 CUDA 13.2 的 PyTorch 软件源。若本机 CUDA 环境不同，请按 PyTorch 官方安装说明调整 `pyproject.toml` 中的 PyTorch 软件源配置。

## 快速开始

运行 FedAvg：

```bash
uv run main.py --algo fedavg --dataset cifar10 --model cnn \
  --num-clients 20 --num-rounds 1000 --devices 0:4
```

`--devices` 用 `GPU编号:worker数量` 指定 worker 分配，例如 `0:2,1:4` 表示 GPU 0 上运行 2 个 worker、GPU 1 上运行 4 个 worker。服务端默认使用 GPU 0，可通过 `--service-device` 指定。

快速检查算法能否启动时，可使用 `--test`。该模式固定运行 10 轮，并将结果标记为 `test`：

```bash
uv run main.py --algo fedavg --test
```

## 常用命令

### 数据划分

默认采用 Dirichlet 划分。使用 IID 划分：

```bash
uv run main.py --algo fedavg --partition iid
```

调整 Dirichlet 参数（数值越小，客户端间标签分布通常越不均衡）：

```bash
uv run main.py --algo fedavg --partition dirichlet --dir 0.1
```

病态划分需指定每个客户端的类别数：

```bash
uv run main.py --algo fedavg --partition pathological --classes-per-client 2
```

数据集会在首次运行时下载到 `datasets/`。可通过 `--data-root` 更改数据目录。

### 多次试验

同一 seed 运行 5 个独立试次：

```bash
uv run main.py --algo fedavg --seed 42 --times 5
```

只运行指定试次（可用于补跑）：

```bash
uv run main.py --algo fedavg --seed 42 --trials 2,4
```

### 结果查询与比较

```bash
# 列出已完成的运行
uv run main.py list

# 查看运行状态及参数
uv run main.py list --status all

# 查看单次运行详情
uv run main.py show <运行ID>

# 比较多个算法
uv run main.py compare --algo fedavg,fedprox,moon

# 比较同一算法的一项参数取值
uv run main.py compare --algo fedprox --vary mu

# 聚合多个试次
uv run main.py aggregate --algo fedavg --trials 1,2,3,4,5
```

运行结果默认保存到 `results/`，每次运行包含 `args.json`、`run.log`、`metrics.csv` 和 `checkpoints/`；`results/results.sqlite` 保存运行摘要。通过 `--result-root` 可更改输出目录。测试运行不纳入默认结果比较和聚合，可用 `list --status test` 单独查看。

## 批量实验与分析

`uv run main.py group` 按 `main.py` 中的 `GROUP_COMMANDS` 执行批量实验计划。计划中每组的命令依次运行，不同组并行执行；如需调整实验内容、公共参数或 GPU 分配，请编辑该配置。

Jupyter Notebook 示例位于 `notebooks/`，覆盖 GFL、PFL 和 DFL 的结果分析。启动 Jupyter：

```bash
uv run --with jupyterlab jupyter lab
```

## 项目结构

```text
algo/       联邦学习算法实现，按 GFL/PFL/DFL 分类
core/       训练编排、模型、数据、进程管理与结果存储
notebooks/  实验结果分析 Notebook
datasets/   下载的数据集（运行时生成）
results/    实验结果、日志、checkpoint 与 SQLite 摘要（运行时生成）
main.py     训练、批处理和结果查询统一入口
```
