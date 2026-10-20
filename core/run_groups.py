import argparse
import shlex
import traceback
from collections.abc import Callable


def algorithm_name(arguments: list[str]) -> str:
    """从命令参数中提取算法名，用于显示执行进度。"""
    for index, argument in enumerate(arguments):
        if argument == "--algo" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--algo="):
            return argument.removeprefix("--algo=")
    raise ValueError("每条命令必须通过 --algo 或 --algo=<名称> 指定算法")


def run_groups(
    common_args_text: str,
    command_texts: list[str],
    argv: list[str] | None = None,
    *,
    run_command: Callable[[list[str]], int],
) -> int:
    """在当前进程按顺序运行共享参数下的算法命令。"""
    argparse.ArgumentParser(description="按顺序运行实验命令").parse_args(argv)
    if not isinstance(common_args_text, str):
        raise TypeError("GROUP_COMMON_ARGS 必须是命令字符串")
    if (
        not isinstance(command_texts, list)
        or not command_texts
        or not all(isinstance(command, str) for command in command_texts)
    ):
        raise ValueError("GROUP_COMMANDS 必须是非空字符串列表")

    common_args = shlex.split(common_args_text)
    commands = []
    for command_text in command_texts:
        command = shlex.split(command_text)
        if not command:
            raise ValueError("GROUP_COMMANDS 不能包含空命令")
        algorithm_args = [*common_args, *command]
        commands.append((algorithm_name(algorithm_args), algorithm_args))

    failures: list[tuple[int, str, int]] = []
    for index, (algorithm, algorithm_args) in enumerate(commands, start=1):
        print(
            f"\n命令 {index}/{len(commands)}：{shlex.join(algorithm_args)}",
            flush=True,
        )
        try:
            return_code = run_command(algorithm_args)
        except Exception:
            traceback.print_exc()
            return_code = 1
        status = "完成" if return_code == 0 else f"失败（退出码 {return_code}）"
        print(f"命令 {index}/{len(commands)} {algorithm} {status}", flush=True)
        if return_code != 0:
            failures.append((index, algorithm, return_code))

    if failures:
        print("\n失败的命令：", flush=True)
        for index, algorithm, return_code in failures:
            print(
                f"  命令 {index}（{algorithm}）：退出码 {return_code}",
                flush=True,
            )
    else:
        print(f"\n全部 {len(commands)} 条命令均已完成。", flush=True)
    return 1 if failures else 0
