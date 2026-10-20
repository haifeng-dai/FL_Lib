"""FL_MP 统一命令入口和通用运行编排。"""

import argparse
import copy
import signal
import sys
import traceback
from datetime import UTC, datetime
from types import ModuleType

import torch

from algo import load_algorithm
from core.config import (
    add_shared_arguments,
    parse_trials,
    prepare_arguments,
    set_seed,
)
from core.run_groups import run_groups
from core.save import (
    configure_logging,
    create_run,
    record_running,
    record_summary,
)
from core.save.cli import run_results

# 共享实验条件和有序算法命令由主进程依次执行。
GROUP_COMMON_ARGS = (
    "--dataset cifar10 --partition dirichlet --dir 0.1 --times 5 --service-device 0"
)
GROUP_COMMANDS: list[str] = [
    "--algo fedavg",
    "--algo feddyn",
    "--algo moon",
    "--algo scaffold",
    "--algo fedprox",
    "--algo fedproc",
    "--algo fedfm",
]


def handle_termination(signum, _) -> None:
    """将终止信号转换为可记录并清理资源的退出。"""
    raise SystemExit(128 + signum)


def run_once(algorithm: ModuleType, args: argparse.Namespace) -> None:
    """以一个明确试次独立运行算法。"""
    torch.multiprocessing.set_sharing_strategy("file_system")
    set_seed(args.seed)
    started = datetime.now(UTC)
    paths = create_run(
        args,
        client_metric="global_acc" if algorithm.Server.pfl else "client_macro_acc",
        extra_metric_columns=getattr(algorithm.Server, "extra_metric_columns", ()),
    )
    record_running(paths, args, started)
    logger = configure_logging(paths, args.log_level)
    args._startup_log_file = str(paths.log_file)
    args._startup_log_lock = torch.multiprocessing.get_context("spawn").Lock()
    previous_sigterm = signal.signal(signal.SIGTERM, handle_termination)
    server = None
    status = "failed"
    try:
        server = algorithm.Server(args)
        server.fit(paths, logger)
        status = "test" if getattr(args, "test_mode", False) else "completed"
    finally:
        ended = datetime.now(UTC)
        accuracies = [] if server is None else server.accuracies
        try:
            if status == "failed":
                logger.exception("运行失败")
            record_summary(
                paths,
                args,
                started,
                ended,
                status,
                max(accuracies, default=0.0),
                accuracies[-1] if accuracies else 0.0,
            )
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.flush()
                handler.close()


def run_algorithm(argv: list[str] | None = None) -> int:
    """解析参数并按框架级试次独立运行算法。"""
    argv = sys.argv[1:] if argv is None else argv
    try:
        algorithm = load_algorithm(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    parser = argparse.ArgumentParser(description=f"FL_MP {algorithm.__name__}")
    add_shared_arguments(parser, algorithm)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    expected_name = algorithm.__name__.rsplit(".", maxsplit=1)[-1]
    if args.algo.lower() != expected_name:
        raise ValueError(f"算法模块 {expected_name} 与 --algo={args.algo} 不一致")
    prepare_arguments(args)
    failed_trials: list[int] = []
    for trial in parse_trials(args):
        trial_args = copy.copy(args)
        trial_args.trial = trial
        print(f"\n============= 第 {trial} 次实验 =============", flush=True)
        try:
            run_once(algorithm, trial_args)
        except Exception:  # noqa: BLE001 - 单个试次失败不应中断其余指定试次
            failed_trials.append(trial)
            traceback.print_exc()
    if failed_trials:
        print("失败的实验编号：" + ", ".join(map(str, failed_trials)), flush=True)
        return 1
    return 0


def main() -> int:
    """按首个参数分发结果、批处理或算法训练命令。"""
    command = sys.argv[1] if len(sys.argv) > 1 else None
    match command:
        case "list" | "show" | "compare" | "aggregate":
            return run_results()
        case "group":
            previous_sigterm = signal.signal(signal.SIGTERM, handle_termination)
            try:
                return run_groups(
                    GROUP_COMMON_ARGS,
                    GROUP_COMMANDS,
                    sys.argv[2:],
                    run_command=run_algorithm,
                )
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
        case _:
            return run_algorithm(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
