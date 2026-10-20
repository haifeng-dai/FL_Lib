import argparse
import logging
import signal
from datetime import UTC, datetime
from types import ModuleType
from typing import Any, Self

import torch

from .config import set_seed
from .save import configure_logging, create_run, record_running, record_summary
from .save.artifacts import RunPaths


def handle_termination(signum, _) -> None:
    """将终止信号转换为可记录并清理资源的退出。"""
    raise SystemExit(128 + signum)


class RunSession:
    """管理单次运行的初始化、状态记录、信号和日志生命周期。"""

    def __init__(self, args: argparse.Namespace, algorithm: ModuleType) -> None:
        self.args = args
        self.algorithm = algorithm
        self.started: datetime | None = None
        self.paths: RunPaths | None = None
        self.logger: logging.Logger | None = None
        self.server: Any = None
        self._previous_sigterm = None

    def __enter__(self) -> Self:
        torch.multiprocessing.set_sharing_strategy("file_system")
        set_seed(self.args.seed)
        self.started = datetime.now(UTC)
        self.paths = create_run(
            self.args,
            client_metric=(
                "global_acc"
                if getattr(self.algorithm.Server, "pfl", False)
                else "client_macro_acc"
            ),
            extra_metric_columns=getattr(
                self.algorithm.Server, "extra_metric_columns", ()
            ),
        )
        record_running(self.paths, self.args, self.started)
        self.logger = configure_logging(self.paths, self.args.log_level)
        self.args._startup_log_file = str(self.paths.log_file)
        self.args._startup_log_lock = torch.multiprocessing.get_context("spawn").Lock()
        self._previous_sigterm = signal.signal(signal.SIGTERM, handle_termination)
        return self

    def fit(self, server: Any) -> None:
        """记录服务端实例并启动训练。"""
        if self.paths is None or self.logger is None:
            raise RuntimeError("RunSession 尚未启动")
        self.server = server
        server.fit(self.paths, self.logger)

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if self.paths is not None and self.started is not None:
                if exc_type is not None and self.logger is not None:
                    self.logger.exception("运行失败")
                accuracies = (
                    [] if self.server is None else self.server.accuracies
                )
                if exc_type is not None:
                    status = "failed"
                elif getattr(self.args, "test_mode", False):
                    status = "test"
                else:
                    status = "completed"
                record_summary(
                    self.paths,
                    self.args,
                    self.started,
                    datetime.now(UTC),
                    status,
                    max(accuracies, default=0.0),
                    accuracies[-1] if accuracies else 0.0,
                )
        finally:
            if self._previous_sigterm is not None:
                signal.signal(signal.SIGTERM, self._previous_sigterm)
            if self.logger is not None:
                for handler in self.logger.handlers[:]:
                    self.logger.removeHandler(handler)
                    handler.flush()
                    handler.close()
        return False
