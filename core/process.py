import argparse
import math
import os
import queue
import traceback
from collections.abc import Mapping
from datetime import datetime
from time import perf_counter
from typing import Any, cast

import torch
import torch.multiprocessing as mp

from .protocol import (
    ClientHook,
    ClientResult,
    ClientTask,
    EvaluationResult,
    EvaluationTask,
)


def startup_trace(
    args: argparse.Namespace, stage: str, worker_id: int | None = None
) -> None:
    """仅在 DEBUG 模式输出可跨进程比较的启动时间戳。"""
    if getattr(args, "log_level", "INFO") != "DEBUG":
        return
    log_file = getattr(args, "_startup_log_file", None)
    log_lock = getattr(args, "_startup_log_lock", None)
    if log_file is None or log_lock is None:
        return
    worker = "main" if worker_id is None else f"worker={worker_id}"
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    message = (
        f"{timestamp} | DEBUG | [startup t={perf_counter():.6f} "
        f"pid={os.getpid()} {worker}] {stage}\n"
    )
    with log_lock, open(log_file, "a", encoding="utf-8") as file:
        file.write(message)
        file.flush()


def _worker(
    worker_id: int,
    device_name: str,
    hook_type: type[ClientHook],
    args: argparse.Namespace,
    num_classes: int,
    train_sets: Mapping[int, Any],
    test_sets: Mapping[int, Any] | None,
    inbox: Any,
    outbox: Any,
) -> None:
    """创建钩子实例并串行执行调度给本 Worker 的任务。"""
    startup_trace(args, "worker_entry", worker_id)
    try:
        device = torch.device(device_name)
        startup_trace(args, f"cuda_set_device_begin device={device_name}", worker_id)
        torch.cuda.set_device(device)
        startup_trace(args, "cuda_set_device_done", worker_id)
        startup_trace(args, "client_init_begin", worker_id)
        hook = hook_type(args, device, num_classes)
        startup_trace(args, "client_init_done", worker_id)
        hook.set_data(train_sets, test_sets)
        startup_trace(args, "set_data_done", worker_id)
    except BaseException:  # noqa: BLE001 -- 初始化失败必须传给主进程。
        outbox.put((worker_id, None, None, traceback.format_exc()))
        return

    while (task := inbox.get()) is not None:
        try:
            if isinstance(task, ClientTask):
                result = hook.run(task)
                _validate_training_result(task, result)
            else:
                result = hook.evaluate(task)
                _validate_evaluation_result(task, result)
            outbox.put((worker_id, task.client_id, result, None))
        except BaseException:  # noqa: BLE001 -- 钩子失败必须传给主进程。
            outbox.put((worker_id, task.client_id, None, traceback.format_exc()))
        finally:
            torch.cuda.empty_cache()


def _validate_training_result(task: ClientTask, result: ClientResult) -> None:
    """在 Worker 内尽早验证算法钩子的基础协议。"""
    if result.client_id != task.client_id:
        raise ValueError("ClientHook 返回的 client_id 与任务不一致")
    if not math.isfinite(result.loss):
        raise FloatingPointError(
            f"客户端 {task.client_id} 返回非有限训练损失：{result.loss}"
        )
    for tensor in result.state.values():
        if tensor.device.type != "cpu" or tensor.requires_grad:
            raise ValueError("ClientHook 必须返回独立的 CPU 模型状态")


def _validate_evaluation_result(task: EvaluationTask, result: EvaluationResult) -> None:
    """验证评估结果的客户端身份和计数范围。"""
    if result.client_id != task.client_id:
        raise ValueError("ClientHook 返回的 client_id 与评估任务不一致")
    if (
        result.correct < 0
        or result.num_samples < 0
        or result.correct > result.num_samples
    ):
        raise ValueError("评估结果的正确数或样本数无效")


class PersistentClientPool:
    """复用 Worker 的 spawn 进程池；不包含任何算法训练逻辑。"""

    def __init__(
        self,
        devices: list[str],
        hook_type: type[ClientHook],
        args: argparse.Namespace,
        num_classes: int,
        train_sets: Mapping[int, Any],
        test_sets: Mapping[int, Any] | None,
    ):
        self.context = mp.get_context("spawn")
        self.inboxes = [self.context.Queue() for _ in devices]
        self.outbox = self.context.Queue()
        self.processes = [
            self.context.Process(
                target=_worker,
                args=(
                    worker_id,
                    device,
                    hook_type,
                    args,
                    num_classes,
                    train_sets,
                    test_sets,
                    self.inboxes[worker_id],
                    self.outbox,
                ),
            )
            for worker_id, device in enumerate(devices)
        ]
        for worker_id, process in enumerate(self.processes):
            started = perf_counter()
            startup_trace(args, f"process_start_begin worker={worker_id}")
            process.start()
            startup_trace(
                args,
                f"process_start_done worker={worker_id} pid={process.pid} "
                f"elapsed={perf_counter() - started:.3f}s",
            )

    def run(self, tasks: list[ClientTask]) -> dict[int, ClientResult]:
        """调度任务；任一 Worker 错误都会使本轮立即失败。"""
        return cast(dict[int, ClientResult], self._run(tasks, "训练"))

    def run_affined(self, tasks: list[ClientTask]) -> dict[int, ClientResult]:
        """按客户端编号固定 Worker 调度，供持有客户端跨轮状态的算法使用。"""
        if len({task.client_id for task in tasks}) != len(tasks):
            raise ValueError("固定调度不允许同一客户端在一轮中出现多次")
        queues: list[list[ClientTask]] = [[] for _ in self.inboxes]
        for task in tasks:
            queues[task.client_id % len(self.inboxes)].append(task)
        return cast(
            dict[int, ClientResult], self._run_worker_queues(queues, "固定调度训练")
        )

    def evaluate(self, tasks: list[EvaluationTask]) -> dict[int, EvaluationResult]:
        """复用同一批 Worker 并发执行客户端私有测试集评估。"""
        return cast(dict[int, EvaluationResult], self._run(tasks, "评估"))

    def _run(self, tasks, action: str):
        """调度同一协议类型的任务；任一 Worker 错误都会立即失败。"""
        self._check_workers_alive(action)
        results: dict[int, Any] = {}
        next_task = 0
        active = 0
        pending_workers: dict[int, int] = {}
        for worker_id in range(min(len(self.inboxes), len(tasks))):
            self.inboxes[worker_id].put(tasks[next_task])
            pending_workers[worker_id] = tasks[next_task].client_id
            next_task += 1
            active += 1
        while active:
            worker_id, client_id, result, error = self._get_result(
                pending_workers, action
            )
            active -= 1
            pending_workers.pop(worker_id, None)
            if error is not None:
                raise RuntimeError(f"worker {worker_id} {action}失败：\n{error}")
            results[client_id] = result
            if next_task < len(tasks):
                self.inboxes[worker_id].put(tasks[next_task])
                pending_workers[worker_id] = tasks[next_task].client_id
                next_task += 1
                active += 1
        return results

    def _run_worker_queues(self, queues, action: str):
        """让每个 Worker 串行消费自己的任务队列。"""
        self._check_workers_alive(action)
        results: dict[int, Any] = {}
        positions = [0] * len(queues)
        active = 0
        pending_workers: dict[int, int] = {}
        for worker_id, tasks in enumerate(queues):
            if tasks:
                self.inboxes[worker_id].put(tasks[0])
                pending_workers[worker_id] = tasks[0].client_id
                positions[worker_id] = 1
                active += 1
        while active:
            worker_id, client_id, result, error = self._get_result(
                pending_workers, action
            )
            active -= 1
            pending_workers.pop(worker_id, None)
            if error is not None:
                raise RuntimeError(f"worker {worker_id} {action}失败：\n{error}")
            results[client_id] = result
            if positions[worker_id] < len(queues[worker_id]):
                self.inboxes[worker_id].put(queues[worker_id][positions[worker_id]])
                pending_workers[worker_id] = queues[worker_id][
                    positions[worker_id]
                ].client_id
                positions[worker_id] += 1
                active += 1
        return results

    def _get_result(self, pending_workers: dict[int, int], action: str):
        """等待 worker 结果，并检测未能回报结果便退出的子进程。"""
        while True:
            try:
                return self.outbox.get(timeout=1)
            except queue.Empty:
                self._check_workers_alive(action, pending_workers)

    def _check_workers_alive(
        self, action: str, pending_workers: dict[int, int] | None = None
    ) -> None:
        """检测异常退出的 worker，避免任务队列永久等待其结果。"""
        dead_workers = [
            (worker_id, process)
            for worker_id, process in enumerate(self.processes)
            if not process.is_alive()
        ]
        if not dead_workers:
            return
        details = []
        for worker_id, process in dead_workers:
            client_id = (pending_workers or {}).get(worker_id)
            task = f"，客户端 {client_id} 的任务未完成" if client_id is not None else ""
            details.append(f"worker {worker_id} (exit code {process.exitcode}){task}")
        raise RuntimeError(f"{action}期间 worker 异常退出：" + "; ".join(details))

    def close(self, *, terminate: bool = False) -> None:
        """通知 Worker 退出并关闭全部 IPC 队列。"""
        forced_termination = terminate
        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for inbox in self.inboxes:
                inbox.put(None)
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                forced_termination = True
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                forced_termination = True
                process.join()
        for queue in [*self.inboxes, self.outbox]:
            if forced_termination:
                queue.cancel_join_thread()
            queue.close()
