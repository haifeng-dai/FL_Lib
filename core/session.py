import argparse
import logging
import signal
from types import ModuleType
from typing import Any, Self

import torch

from .config import set_seed
from .result import Results


def handle_termination(signum, _) -> None:
    """将终止信号转换为可记录并清理资源的退出。"""
    raise SystemExit(128 + signum)


class RunSession:
    """管理单次运行的初始化、状态记录、信号和日志生命周期。"""

    def __init__(self, args: argparse.Namespace, algorithm: ModuleType) -> None:
        self.args = args
        self.algorithm = algorithm
        self.results = Results()
        self.logger: logging.Logger | None = None
        self.server: Any = None
        self._previous_sigterm = None

    def __enter__(self) -> Self:
        torch.multiprocessing.set_sharing_strategy("file_system")
        set_seed(self.args.seed)
        algorithm_script = getattr(self.algorithm, "__file__", None)
        if algorithm_script is None:
            raise RuntimeError(f"算法模块 {self.algorithm.__name__} 没有关联的脚本文件")
        self.results.start(
            self.args,
            algorithm_script=algorithm_script,
            client_metric=(
                "global_acc"
                if getattr(self.algorithm.Server, "pfl", False)
                else "client_macro_acc"
            ),
            extra_metric_columns=getattr(
                self.algorithm.Server, "extra_metric_columns", ()
            ),
        )
        self._previous_sigterm = signal.signal(signal.SIGTERM, handle_termination)
        self.logger = self.results.configure_logging(self.args.log_level)
        self.args._startup_log_file = str(self.results.startup_log_file)
        self.args._startup_log_lock = torch.multiprocessing.get_context("spawn").Lock()
        return self

    def fit(self, server: Any) -> None:
        """记录服务端实例并启动训练。"""
        if self.logger is None:
            raise RuntimeError("RunSession 尚未启动")
        self.server = server
        server.fit(self.results, self.logger)

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if self.results.run_id is not None:
                if exc_type is not None and self.logger is not None:
                    self.logger.exception("运行失败")
                if exc_type is not None:
                    status = "failed"
                elif getattr(self.args, "test_mode", False):
                    status = "test"
                else:
                    status = "completed"
                self.results.finish(status)
        finally:
            if self._previous_sigterm is not None:
                signal.signal(signal.SIGTERM, self._previous_sigterm)
        return False
