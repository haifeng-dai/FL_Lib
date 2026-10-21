import argparse
from collections.abc import Sequence


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查询 FL_Lib 实验结果")
    parser.add_argument("command", nargs="?", choices=("list", "show"), default="list")
    parser.add_argument("run_id", nargs="?", help="show 命令的运行 ID")
    parser.add_argument(
        "-a", "--algo", type=_csv_values, help="算法名，可用逗号分隔多个算法"
    )
    parser.add_argument(
        "--dataset", type=_csv_values, help="数据集名称，可用逗号分隔多个数据集"
    )
    parser.add_argument(
        "--status",
        type=_csv_values,
        help=(
            "运行状态，可用逗号分隔多个状态：running、completed、failed、test；"
            "all 表示全部。不指定筛选条件时显示最近完成的 50 条和全部运行中的记录"
        ),
    )
    args = parser.parse_args(argv)
    args.status_explicit = args.status is not None
    valid_statuses = {"running", "completed", "failed", "test", "all"}
    if args.status and any(status not in valid_statuses for status in args.status):
        parser.error("--status 仅支持 running、completed、failed、test、all")
    if args.status and "all" in args.status and len(args.status) > 1:
        parser.error("--status all 不能与其他状态同时使用")
    if args.status == ("all",):
        args.status = None
    if args.command == "show" and not args.run_id:
        parser.error("show 命令需要提供 run_id")
    if args.command == "list" and args.run_id:
        parser.error("run_id 仅适用于 show 命令")
    if args.command == "show" and (args.algo or args.dataset or args.status):
        parser.error("--algo、--dataset 和 --status 仅适用于 list 命令")
    return args
