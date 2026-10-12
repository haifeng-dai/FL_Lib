import abc
import argparse
import random
import sys
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Subset

from core.save import append_metrics
from core.save.artifacts import MetricsRow

from .config import parse_devices, set_seed
from .data import load_federated_data, make_loader
from .model import Model, build_model
from .process import PersistentClientPool, startup_trace
from .protocol import (
    ClientHook,
    ClientResult,
    ClientTask,
    EvaluationResult,
    EvaluationTask,
)
from .state import clone_state


class ProgressBar:
    """无额外依赖的服务端终端进度条。"""

    def __init__(self, algorithm: str, total: int):
        self.algorithm = algorithm.upper()
        self.total = total
        self.current = 0

    def update(
        self,
        loss: float,
        accuracy: float,
        max_accuracy: float,
        round_elapsed: float,
        total_elapsed: float,
        extra_fields: dict[str, str | int | float] | None = None,
    ) -> None:
        """刷新当前通信轮进度与基础指标。"""
        self.current += 1
        average_elapsed = total_elapsed / self.current
        remaining = average_elapsed * (self.total - self.current)
        extras = (
            ""
            if not extra_fields
            else " "
            + " ".join(f"{name}={value}" for name, value in extra_fields.items())
        )
        text = (
            f"\r{self.algorithm} {self.current}/{self.total} "
            f"loss={loss:.4f} acc={accuracy:.2f}% max_acc={max_accuracy:.2f}% "
            f"elapsed={format_duration(total_elapsed)} eta={format_duration(remaining)} "
            f"round_time={round_elapsed:.2f}s{extras}"
        )
        sys.stderr.write(text)
        sys.stderr.flush()

    def close(self) -> None:
        """结束进度行。"""
        if self.current:
            sys.stderr.write("\n")
            sys.stderr.flush()


class BaseClient(ClientHook):
    """算法客户端基础类：构造时建立模型、设备和本地训练参数。"""

    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device,
        num_classes: int,
    ) -> None:
        """读取构造参数并保存客户端实际需要的明确字段。"""
        self.device = device
        self.num_classes = num_classes
        self.model_name: str = args.model
        self.dataset_name: str = args.dataset
        self.feature_dim: int = args.feature_dim
        self.batch_size: int = args.batch_size
        self.num_epochs: int = args.num_epochs
        self.lr: float = args.lr
        self.momentum: float = args.momentum
        self.weight_decay: float = args.weight_decay

        self.model = build_model(
            self.model_name,
            self.dataset_name,
            self.num_classes,
            self.feature_dim,
        )
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def set_data(
        self, train_sets: dict[int, Subset], test_sets: dict[int, Subset] | None = None
    ) -> None:
        """绑定客户端训练数据及可选的个性化测试数据。"""
        self.train_sets = train_sets
        self.test_sets = test_sets

    def run(self, task: ClientTask) -> ClientResult:
        """统一加载任务状态后执行算法注入的本地训练。"""
        set_seed(task.seed)
        self.model.load_state_dict(task.state)
        self.model.to(self.device)
        return self.train(task)

    @abc.abstractmethod
    def train(self, task: ClientTask) -> ClientResult:
        """执行算法特有的本地训练。"""
        raise NotImplementedError

    def train_loader(self, client_id: int):
        """创建指定客户端的训练 DataLoader。空数据集按约定自然报错。"""
        return make_loader(self.train_sets[client_id], self.batch_size, shuffle=True)

    @torch.no_grad()
    def evaluate(self, task: EvaluationTask) -> EvaluationResult:
        """在指定客户端私有测试集上加载状态并计算正确预测数。"""
        if self.test_sets is None:
            raise RuntimeError("全局算法客户端未绑定个性化测试集")
        self.model.load_state_dict(task.state)
        self.model.to(self.device).eval()
        correct, samples = 0, 0
        for inputs, targets in make_loader(
            self.test_sets[task.client_id], self.batch_size
        ):
            predictions = (
                self.model(inputs.to(self.device, non_blocking=True))
                .argmax(dim=1)
                .cpu()
            )
            correct += (predictions == targets).sum().item()
            samples += len(targets)
        return EvaluationResult(task.client_id, correct, samples)

    def copy_model(self, state: dict[str, torch.Tensor]) -> Model:
        """重建并冻结一个加载指定状态的参考模型。"""
        model = build_model(
            self.model_name,
            self.dataset_name,
            self.num_classes,
            self.feature_dim,
        )
        model.load_state_dict(state)
        model.to(self.device)
        model.eval().requires_grad_(False)
        return model

    def build_optimizer(self, model: Model | None = None) -> torch.optim.Optimizer:
        """构建默认 SGD；使用其他优化器的算法可覆写本方法。"""
        model_opt = self.model if model is None else model
        return torch.optim.SGD(
            model_opt.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def train_supervised(self, client_id: int) -> float:
        """执行通用本地监督训练，返回每个 batch 的平均损失。"""
        self.model.train()
        optimizer = self.build_optimizer()
        total_loss, batches = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(client_id):
                optimizer.zero_grad(set_to_none=True)
                output = self.model(inputs.to(self.device, non_blocking=True))
                loss = self.ce_loss(output, targets.to(self.device, non_blocking=True))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1
        return total_loss / batches

    def result(
        self, task: ClientTask, loss: float, payload: Any = None
    ) -> ClientResult:
        """生成符合跨进程约束的标准客户端结果。"""
        return ClientResult(
            client_id=task.client_id,
            state=clone_state(self.model.state_dict()),
            num_samples=len(self.train_sets[task.client_id]),
            loss=loss,
            payload=payload,
        )


def format_duration(seconds: float) -> str:
    """将秒数格式化为适合终端展示的 HH:MM:SS。"""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class BaseServer(abc.ABC):
    """算法服务端基础类：构造时建立通用联邦训练资源。"""

    client_class: type[BaseClient] | None = None
    pfl = False

    def __init__(self, args: argparse.Namespace):
        """读取运行参数并建立数据、模型和持久 Worker 池。"""
        if self.client_class is None:
            raise TypeError("算法必须设置 client_class")
        self.eval_clients: bool = getattr(args, "eval_clients", False)
        self.algorithm = args.algo
        self.seed = args.seed
        self.num_rounds = args.num_rounds
        self.join_ratio = args.join_ratio
        self.batch_size = args.batch_size
        self.check_round = args.check_round
        self.service_device = args.service_device
        self.device = torch.device(f"cuda:{self.service_device}")
        startup_trace(args, "load_federated_data_begin")
        self.train_sets, self.test_sets, self.num_classes = load_federated_data(
            args.dataset,
            args.data_root,
            args.test_ratio,
            args.partition,
            args.num_clients,
            args.dir,
            args.classes_per_client,
            args.seed,
        )
        startup_trace(args, "load_federated_data_done")
        self.test_set = ConcatDataset(list(self.test_sets.values()))
        self.model = build_model(
            args.model,
            args.dataset,
            self.num_classes,
            args.feature_dim,
            args.pre_train,
        )
        self.pool = PersistentClientPool(
            parse_devices(args.devices),
            self.client_class,
            args,
            self.num_classes,
            self.train_sets,
            self.test_sets if self.pfl or self.eval_clients else None,
        )
        self.random = random.Random(self.seed)
        self.client_task_batch = 0
        self.selected: list[int] = []
        self.accuracies: list[float] = []
        self.num_clients = args.num_clients
        self.client_states = self.replicated_model_states() if self.pfl else {}
        # 上传缓存独立于训练状态；未上传者暂用初始模型，未参与者保留最近上传。
        self.uploaded_states = (
            self.replicated_model_states() if self.eval_clients else {}
        )
        self.client_eval_metrics: dict[str, float] = {}
        self.client_eval_history: list[dict[str, float]] = []
        self.personalized_client_accuracies: dict[int, float] = {}

    def fit(self, paths, logger) -> None:
        """运行标准通信轮并记录基础指标。"""
        progress = ProgressBar(self.algorithm, self.num_rounds)
        fit_started = perf_counter()
        try:
            for round_index in range(1, self.num_rounds + 1):
                started = perf_counter()
                loss, accuracy = self.run_round()
                elapsed = perf_counter() - started
                self.accuracies.append(accuracy)
                metrics: MetricsRow = {
                    "round": round_index,
                    "train_loss": loss,
                    "accuracy": accuracy,
                    "elapsed_seconds": elapsed,
                }
                extra_metrics = self.additional_metrics()
                metrics.update(extra_metrics)
                if self.eval_clients:
                    if self.pfl:
                        metrics["global_acc"] = self.client_eval_metrics["global_acc"]
                    else:
                        metrics["client_macro_acc"] = self.client_eval_metrics[
                            "client_macro_acc"
                        ]
                append_metrics(paths, metrics)
                logger.info(
                    "round=%d | loss=%.6f | accuracy=%.4f | elapsed=%.3fs",
                    round_index,
                    loss,
                    accuracy,
                    elapsed,
                )
                for name, value in extra_metrics.items():
                    logger.info("%s=%.4f", name, value)
                if self.eval_clients:
                    name = "global_acc" if self.pfl else "client_macro_acc"
                    logger.info(
                        "%s=%.4f",
                        name,
                        self.client_eval_metrics[name],
                    )
                progress.update(
                    loss,
                    accuracy,
                    max(self.accuracies),
                    elapsed,
                    perf_counter() - fit_started,
                    self.progress_fields(),
                )
                if round_index % self.check_round == 0:
                    self.save_checkpoint(paths, round_index)
        except BaseException:
            self.pool.close(terminate=True)
            raise
        finally:
            progress.close()
        self.pool.close()

    def run_round(self) -> tuple[float, float]:
        """执行标准选择、训练、聚合、评估流程。"""
        self.select_clients()
        results = self.run_clients()
        self.apply_results(results)
        loss = sum(results[client_id].loss for client_id in self.selected) / len(
            self.selected
        )
        return loss, self.evaluate()

    def training_payloads(self) -> dict[int, dict] | None:
        """返回本轮下发给各客户端的算法私有训练数据。"""
        return None

    def training_states(self) -> dict[int, dict[str, torch.Tensor]] | None:
        """返回本轮下发的模型状态；默认使用全局或客户端私有状态。"""
        return None

    def select_clients(self) -> None:
        """按 join_ratio 无放回随机选择本轮客户端。"""
        count = max(1, int(self.num_clients * self.join_ratio))
        self.selected = sorted(self.random.sample(range(self.num_clients), count))

    def build_tasks(self) -> list[ClientTask]:
        """为已选择客户端构造共享全局模型状态的训练任务。"""
        payloads = self.training_payloads()
        states = self.training_states()
        return [
            ClientTask(
                client_id=client_id,
                state=clone_state(
                    states[client_id]
                    if states is not None
                    else (
                        self.client_states[client_id]
                        if self.pfl
                        else self.model.state_dict()
                    )
                ),
                seed=self.client_seed(client_id),
                payload=None if payloads is None else payloads.get(client_id),
            )
            for client_id in self.selected
        ]

    def client_seed(self, client_id: int) -> int:
        """按任务批次和客户端编号派生可复现的独立随机种子。"""
        return self.seed + self.client_task_batch * self.num_clients + client_id

    def run_clients(self, *, cache_uploads: bool = True) -> dict[int, ClientResult]:
        """执行客户端任务；纯提取等辅助阶段可跳过上传模型缓存。"""
        self.client_task_batch += 1
        results = self.pool.run(self.build_tasks())
        if cache_uploads:
            self.update_uploaded_states(results)
        return results

    def update_uploaded_states(self, results: dict[int, ClientResult]) -> None:
        """按需缓存独立 CPU 模型副本，供普通和固定 Worker 调度共同使用。"""
        if self.eval_clients:
            for client_id, result in results.items():
                self.uploaded_states[client_id] = clone_state(result.state)

    def aggregate_weighted(
        self,
        results: dict[int, ClientResult],
        weights: dict[int, float] | None = None,
    ) -> None:
        """按样本数或算法提供的权重聚合客户端模型状态。"""
        if weights is None:
            total = sum(results[client_id].num_samples for client_id in self.selected)
            weights = {
                client_id: results[client_id].num_samples / total
                for client_id in self.selected
            }
        aggregate = clone_state(self.model.state_dict())
        first_client = self.selected[0]
        for name, tensor in aggregate.items():
            if torch.is_floating_point(tensor):
                tensor.zero_()
                for client_id in self.selected:
                    tensor.add_(
                        results[client_id].state[name], alpha=weights[client_id]
                    )
            else:
                tensor.copy_(results[first_client].state[name])
        self.model.load_state_dict(aggregate)

    @torch.no_grad()
    def evaluate_global(self) -> float:
        """在服务端 GPU 上评估全局模型。"""
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        for inputs, targets in make_loader(self.test_set, self.batch_size):
            prediction = self.model(inputs.to(self.device)).argmax(dim=1).cpu()
            correct += (prediction == targets).sum().item()
            total += len(targets)
        return 100 * correct / total

    def evaluate(self) -> float:
        """保留默认 accuracy 口径，启用开关时同时记录全局和客户端指标。"""
        if self.pfl:
            accuracy = self.evaluate_clients()
            if not self.eval_clients:
                return accuracy
            self.client_eval_metrics = {"global_acc": self.evaluate_global()}
        else:
            accuracy = self.evaluate_global()
            if not self.eval_clients:
                return accuracy
            accuracies = self.evaluate_client_models(self.uploaded_states)
            macro = sum(accuracies.values()) / len(accuracies)
            self.client_eval_metrics = {"client_macro_acc": macro}
        self.client_eval_history.append(self.client_eval_metrics.copy())
        return accuracy

    def replicated_model_states(self) -> dict[int, dict[str, torch.Tensor]]:
        """返回由当前模型复制出的各客户端初始状态。"""
        initial = clone_state(self.model.state_dict())
        return {
            client_id: clone_state(initial) for client_id in range(self.num_clients)
        }

    def update_client_states(self, results: dict[int, ClientResult]) -> None:
        """写回本轮参与客户端的完整私有模型状态。"""
        if not self.client_states:
            raise RuntimeError("算法尚未创建客户端私有模型状态")
        for client_id in self.selected:
            self.client_states[client_id] = clone_state(results[client_id].state)

    @torch.no_grad()
    def evaluate_client_models(
        self, states: dict[int, dict[str, torch.Tensor]]
    ) -> dict[int, float]:
        """评估指定客户端模型，返回各客户端准确率。"""
        tasks = [
            EvaluationTask(client_id, clone_state(states[client_id]))
            for client_id, test_set in self.test_sets.items()
            if len(test_set)
        ]
        results = self.pool.evaluate(tasks)
        accuracies: dict[int, float] = {}
        for client_id, result in results.items():
            accuracies[client_id] = 100 * result.correct / result.num_samples
        if not accuracies:
            raise ValueError("客户端模型评估没有可用的客户端测试样本")
        return accuracies

    def evaluate_clients(self) -> float:
        """评估私有训练模型，返回客户端准确率的等权平均。"""
        accuracies = self.evaluate_client_models(self.client_states)
        self.personalized_client_accuracies = accuracies
        return sum(accuracies.values()) / len(accuracies)

    def save_checkpoint(self, paths, round_index: int) -> None:
        """保存所有算法都需要的基础恢复状态。"""
        torch.save(
            {
                "round": round_index,
                "model": clone_state(self.model.state_dict()),
                "accuracies": self.accuracies,
                "python_random_state": self.random.getstate(),
                "torch_random_state": torch.random.get_rng_state(),
                "algorithm_state": self.checkpoint_state(),
            },
            paths.checkpoint_dir / f"round_{round_index}.pt",
        )

    def checkpoint_state(self) -> dict[str, Any]:
        """供算法覆写以保存额外恢复状态。"""
        state: dict[str, Any] = {}
        if self.pfl:
            state["client_states"] = {
                client_id: clone_state(state)
                for client_id, state in self.client_states.items()
            }
        if self.eval_clients:
            state["uploaded_states"] = {
                client_id: clone_state(model_state)
                for client_id, model_state in self.uploaded_states.items()
            }
            state["client_eval_history"] = [
                metrics.copy() for metrics in self.client_eval_history
            ]
        return state

    def progress_fields(self) -> dict[str, Any]:
        """附加另一种模型的准确率，避免重复显示主指标 acc。"""
        if not self.eval_clients:
            return {}
        return {
            name: f"{value:.2f}%" for name, value in self.client_eval_metrics.items()
        }

    def additional_metrics(self) -> dict[str, float]:
        """返回算法需要额外逐轮记录的指标。"""
        return {}

    @abc.abstractmethod
    def apply_results(self, results: dict[int, ClientResult]) -> None:
        """应用客户端结果，例如 FedAvg 聚合或算法特有状态更新。"""
        raise NotImplementedError
