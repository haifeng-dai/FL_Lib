"""FL_Lib 统一命令入口和运行编排。"""

import argparse
import copy
import signal
import sys
import traceback

from algo import load_algorithm
from core import RunSession, handle_termination
from core.cli.commands import run_results
from core.config import (
    add_shared_arguments,
    parse_trials,
    prepare_arguments,
)
from core.run_groups import run_groups

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


def run_algorithm(argv: list[str] | None = None) -> int:
    """解析参数并按框架级试次独立运行算法。"""
    argv = sys.argv[1:] if argv is None else argv
    try:
        algorithm = load_algorithm(argv)
        parser = argparse.ArgumentParser(description=f"FL_Lib {algorithm.__name__}")
        add_shared_arguments(parser, algorithm)
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    prepare_arguments(args)
    failed_trials: list[int] = []
    for trial in parse_trials(args):
        trial_args = copy.copy(args)
        trial_args.trial = trial
        print(f"\n============= 第 {trial} 次实验 =============", flush=True)
        try:
            with RunSession(trial_args, algorithm) as session:
                session.fit(algorithm.Server(trial_args))
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
        case "list" | "show":
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
