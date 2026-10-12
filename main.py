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
    set_seed,
    validate_arguments,
)
from core.run_groups import run_groups
from core.save import (
    configure_logging,
    create_run,
    record_running,
    record_summary,
)
from core.save.cli import run_results

# 直接在这里维护实验计划。各组同时运行，每组内的命令按列表顺序串行执行。
# common_args 是该组所有算法共享的实验条件，用于公平比较。
# commands 只填写算法名与该算法的特定参数；两组都会使用此处指定的全部 GPU。
GROUP_COMMANDS: dict[str, dict[str, str | list[str]]] = {
    "group_a": {
        "common_args": "--dataset cifar10 --partition dirichlet --dir 0.1 --devices 0:5,1:5,2:5,3:5 --service-device 0",
        "commands": [
            "--algo fedavg",
            "--algo feddyn --alpha-coef 0.01",
            "--algo moon --mu 0.01 --tau 0.5",
            "--algo scaffold --global-lr 1.0",
        ],
    },
    "group_b": {
        "common_args": "--dataset cifar10 --partition dirichlet --dir 0.1 --devices 0:5,1:5,2:5,3:5 --service-device 3",
        "commands": [
            "--algo fedprox --mu 0.01",
            "--algo fedproc",
            "--algo fedfm --mu 0.1",
            "--algo fedpln --lambda 80.0",
        ],
    },
}


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
    try:
        server = algorithm.Server(args)
        server.fit(paths, logger)
    except (Exception, KeyboardInterrupt, SystemExit):
        ended = datetime.now(UTC)
        logger.exception("运行失败")
        accuracies = [] if server is None else server.accuracies
        record_summary(
            paths,
            args,
            started,
            ended,
            "failed",
            max(accuracies, default=0.0),
            accuracies[-1] if accuracies else 0.0,
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    ended = datetime.now(UTC)
    record_summary(
        paths,
        args,
        started,
        ended,
        "test" if getattr(args, "test_mode", False) else "completed",
        max(server.accuracies),
        server.accuracies[-1],
    )


def run_algorithm(algorithm: ModuleType) -> int:
    """解析参数并按框架级试次独立运行算法。"""
    parser = argparse.ArgumentParser(description=f"FL_MP {algorithm.__name__}")
    add_shared_arguments(parser, algorithm)
    args = parser.parse_args()
    expected_name = algorithm.__name__.rsplit(".", maxsplit=1)[-1]
    if args.algo.lower() != expected_name:
        raise ValueError(f"算法模块 {expected_name} 与 --algo={args.algo} 不一致")
    validate_arguments(args)
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
                return run_groups(GROUP_COMMANDS, sys.argv[2:])
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
        case _:
            return run_algorithm(load_algorithm())


if __name__ == "__main__":
    raise SystemExit(main())
