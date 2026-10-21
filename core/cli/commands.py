import sqlite3
import sys
from datetime import datetime
from typing import Any

from core.result import ResultsReader, RunQuery

from .display import display_width, print_table, render_accuracy_chart
from .parser import parse_args


def run_results(argv=None) -> int:
    args = parse_args(argv)
    reader = ResultsReader()
    try:
        if args.command == "show":
            _show_run(reader, args.run_id)
        else:
            _list_runs(reader, args)
    except (OSError, sqlite3.Error, ValueError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 0


def _list_runs(reader: ResultsReader, args) -> None:
    query_args = {"dataset": args.dataset} if args.dataset else {}
    if args.status_explicit:
        runs = reader.query_runs(
            RunQuery(
                algorithms=args.algo,
                statuses=args.status,
                args=query_args,
                apply_defaults=False,
            )
        )
    else:
        running = reader.query_runs(
            RunQuery(
                algorithms=args.algo,
                statuses=("running",),
                args=query_args,
                apply_defaults=False,
            )
        )
        completed = reader.query_runs(
            RunQuery(
                algorithms=args.algo,
                statuses=("completed",),
                args=query_args,
                apply_defaults=False,
            )
        )
        if args.algo is not None or args.dataset is not None:
            runs = running + completed
        else:
            runs = running + completed[:50]

    runs.sort(key=lambda run: (run.started_at, run.run_id))
    if not runs:
        print("没有符合条件的运行。")
        return
    rows = [_run_row(run) for run in runs]
    print_table(
        [
            "运行 ID",
            "算法",
            "状态",
            "数据集",
            "划分",
            "试次",
            "轮次",
            "最佳 acc",
            "最终 acc",
            "耗时",
            "开始时间",
        ],
        rows,
        aligns=[
            "left",
            "left",
            "left",
            "left",
            "left",
            "left",
            "right",
            "right",
            "right",
            "left",
            "left",
        ],
    )


def _show_run(reader: ResultsReader, run_id: str) -> None:
    run = reader.get_run(run_id)
    metrics = reader.read_metrics(run_id)
    points = [
        (int(row["round"]), float(row["accuracy"]))
        for row in metrics
        if row.get("round") is not None and row.get("accuracy") is not None
    ]

    print(f"运行 ID：{run.run_id}")
    print(f"算法：{run.algorithm}    状态：{run.status}")
    duration = _format_duration(run.started_at, run.ended_at, run.status)
    print(
        f"最佳 acc：{run.max_accuracy:.2f}%    "
        f"最终 acc：{run.latest_accuracy:.2f}%    "
        f"耗时：{duration}"
    )
    print("\n参数：")
    parameters = sorted(run.args.items())
    name_width = max((display_width(name) for name, _ in parameters), default=0)
    for name, value in parameters:
        padding = " " * (name_width - display_width(name))
        print(f"  {name}{padding} = {value}")
    print("\n准确率曲线：")
    print(render_accuracy_chart(points))


def _run_row(run) -> list[Any]:
    args = run.args
    partition = args.get("partition", "-")
    if partition == "dirichlet":
        partition = f"Dir({args.get('dir', '-')})"
    elif partition == "pathological":
        partition = f"Pat({args.get('classes_per_client', '-')})"
    elif partition == "iid":
        partition = "IID"

    trial = args.get("trial", "-")
    rounds = args.get("num_rounds", "-")
    progress = f"{run.latest_round}/{rounds}"
    accuracy = (f"{run.max_accuracy:.2f}%", f"{run.latest_accuracy:.2f}%")
    duration = _format_duration(run.started_at, run.ended_at, run.status)
    return [
        run.run_id,
        run.algorithm,
        run.status,
        args.get("dataset", "-"),
        partition,
        trial,
        progress,
        *accuracy,
        duration,
        run.started_at.replace("T", " ")[:19],
    ]


def _format_duration(started_at: str, ended_at: str, status: str = "") -> str:
    if not started_at:
        return "-"
    try:
        start = datetime.fromisoformat(started_at)
        if ended_at:
            end = datetime.fromisoformat(ended_at)
        elif status == "running":
            end = datetime.now().astimezone()
        else:
            return "-"
        seconds = max(0, round((end - start).total_seconds()))
    except (ValueError, TypeError):
        return "-"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
