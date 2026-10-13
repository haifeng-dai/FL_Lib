import argparse
import shutil
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .results import (
    Run as RunRecord,
)
from .results import (
    aggregate_series,
    argument,
    baseline_differences,
    filter_runs,
    required_argument,
    select_latest_comparable,
    select_parameter_comparison,
    select_trials_for_aggregate,
    trial_number,
)
from .results import (
    load_runs as load_result_runs,
)
from .results import (
    read_accuracy_points as read_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = PROJECT_ROOT / "results" / "results.sqlite"
SYMBOLS = ("●", "◆", "■", "▲", "●", "◆", "■", "▲")


@dataclass(frozen=True)
class Curve:
    """一条用于终端预览的准确率曲线。"""

    label: str
    run: RunRecord
    points: list[tuple[int, float]]


def parse_args() -> argparse.Namespace:
    """解析结果浏览命令。"""
    parser = argparse.ArgumentParser(description="浏览 FL_MP 实验结果")
    parser.add_argument(
        "command", nargs="?", choices=("list", "show", "compare", "aggregate")
    )
    parser.add_argument("run_id", nargs="?", help="show 命令要查看的运行 ID")
    parser.add_argument(
        "--database", type=Path, default=DEFAULT_DATABASE, help="SQLite 结果库路径"
    )
    parser.add_argument("-a", "--algo", help="逗号分隔的算法名称")
    parser.add_argument(
        "--status",
        choices=("running", "completed", "failed", "test", "all"),
        help="运行状态筛选；默认仅显示已完成运行，可设为 all 查看全部",
    )
    parser.add_argument("--limit", type=int, help="list 命令最多显示的记录数")
    parser.add_argument("--run", help="compare 命令使用的逗号分隔运行 ID")
    parser.add_argument("--vary", help="compare 单算法时变化的参数名")
    parser.add_argument("--trial", type=int, help="筛选单个试次编号")
    parser.add_argument("--trials", help="逗号分隔的试次编号，例如 1,3,5")
    args = parser.parse_args()
    args.command = args.command or "list"
    if args.status is None:
        args.status = "completed"
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须为正数")
    if args.command == "show" and args.run_id is None:
        parser.error("show 命令需要提供运行 ID")
    if args.command != "show" and args.run_id is not None:
        parser.error("只有 show 命令接受位置运行 ID")
    if args.command == "compare" and not args.run and not args.algo:
        parser.error("compare 命令需要 --run 或 --algo")
    if args.vary and args.command != "compare":
        parser.error("--vary 仅适用于 compare 命令")
    if args.run and args.command != "compare":
        parser.error("--run 仅适用于 compare 命令")
    if args.run and args.vary:
        parser.error("--run 与 --vary 不能同时使用")
    if args.trial is not None and args.trial < 1:
        parser.error("--trial 必须为正数")
    if args.trial is not None and args.trials:
        parser.error("--trial 与 --trials 不能同时使用")
    if args.command == "aggregate" and not args.algo:
        parser.error("aggregate 命令需要 --algo")
    if args.command == "aggregate" and args.run:
        parser.error("aggregate 不支持 --run")
    return args


def parse_csv(value: str | None) -> list[str]:
    """解析逗号分隔参数并去重。"""
    if value is None:
        return []
    values = [item.strip() for item in value.split(",") if item.strip()]
    return list(dict.fromkeys(values))


def parse_trials(value: str | None) -> list[int] | None:
    """解析逗号分隔的正整数试次编号。"""
    if value is None:
        return None
    try:
        trials = [int(item) for item in parse_csv(value)]
    except ValueError as exc:
        raise ValueError("trials 必须是逗号分隔的正整数") from exc
    if not trials or any(trial < 1 for trial in trials):
        raise ValueError("trials 必须是逗号分隔的正整数")
    return trials


def load_runs(
    database: Path, algorithms: list[str] | None = None, status: str = "completed"
) -> list[RunRecord]:
    """命令行的状态快捷入口；数据库和文件读取由公共层完成。"""
    statuses = None if status == "all" else {status}
    return load_result_runs(
        database,
        project_root=PROJECT_ROOT,
        algorithms=algorithms,
        statuses=statuses,
    )


def format_duration(started_at: str, ended_at: str) -> str:
    """格式化 SQLite 记录的运行时长。"""
    try:
        elapsed = datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)
        seconds = max(0, round(elapsed.total_seconds()))
    except ValueError:
        return "-"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def dataset_partition(record: RunRecord) -> str:
    """紧凑显示数据集及数据划分参数。"""
    dataset = argument(record, "dataset")
    partition = argument(record, "partition")
    if partition == "dirichlet":
        partition = f"Dir({argument(record, 'dir')})"
    elif partition == "pathological":
        partition = f"病态({argument(record, 'classes_per_client')})"
    elif partition == "iid":
        partition = "IID"
    return f"{dataset} / {partition}"


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    """输出无依赖的 Unicode 表格。"""
    values = [[str(value) for value in row] for row in rows]
    widths = [display_width(header) for header in headers]
    for row in values:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], display_width(value))
    rule = "-+-".join("-" * width for width in widths)
    print(
        " | ".join(pad_display(header, width) for header, width in zip(headers, widths))
    )
    print(rule)
    for row in values:
        print(
            " | ".join(
                pad_display(value, width)
                for value, width in zip(row, widths, strict=True)
            )
        )


def display_width(value: str) -> int:
    """计算 CJK 宽字符在等宽终端中的显示宽度。"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in value)


def pad_display(value: str, width: int) -> str:
    """按显示宽度而不是 Python 字符数填充表格单元格。"""
    return value + " " * (width - display_width(value))


def list_runs(records: list[RunRecord], limit: int | None) -> None:
    """显示运行摘要表。"""
    recent = records if limit is None else records[:limit]
    shown = list(reversed(recent))
    rows = [
        [
            record.run_id,
            record.algorithm,
            record.status,
            dataset_partition(record),
            argument(record, "num_clients"),
            argument(record, "num_rounds"),
            trial_number(record) or "-",
            f"{record.max_accuracy:.2f} / {record.latest_accuracy:.2f}%",
            format_duration(record.started_at, record.ended_at),
            record.started_at.replace("T", " ")[:19],
        ]
        for record in shown
    ]
    if not rows:
        print("没有符合条件的运行。")
        return
    print_table(
        [
            "运行 ID",
            "算法",
            "状态",
            "数据集 / 划分",
            "客户端",
            "轮数",
            "试次",
            "最佳 / 最终 acc",
            "时长",
            "开始时间",
        ],
        rows,
    )
    issues = [record for record in shown if record.issue]
    if issues:
        print("\n目录检查：")
        for record in issues:
            print(f"- {record.run_id}: {record.issue}")


def print_run(record: RunRecord) -> None:
    """显示单次运行的摘要、参数与准确率曲线。"""
    print(f"运行 ID：{record.run_id}")
    print(f"算法：{record.algorithm}    状态：{record.status}")
    print(
        f"最佳 acc：{record.max_accuracy:.2f}%    最终 acc：{record.latest_accuracy:.2f}%"
    )
    print(f"目录：{record.run_path}")
    if record.issue:
        print(f"目录检查：{record.issue}")
    if record.args is not None:
        print("\n参数：")
        for name, value in sorted(record.args.items()):
            print(f"  {name} = {value}")
    try:
        points = read_metrics(record)
    except ValueError as exc:
        print(f"\n无法预览准确率曲线：{exc}")
        return
    print("\n准确率曲线：")
    print(render_accuracy_chart([Curve(record.algorithm, record, points)]))


def print_baseline_differences(records: list[RunRecord]) -> None:
    """显示显式指定运行之间不同的基础参数。"""
    differences = baseline_differences(records)
    if differences:
        print("基础参数差异：" + ", ".join(differences))


def compare_runs(
    records: list[RunRecord], vary: str | None = None, explicit: bool = False
) -> None:
    """输出比较摘要与多曲线 Unicode 图。"""
    curves: list[Curve] = []
    rows: list[list[Any]] = []
    for record in records:
        try:
            points = read_metrics(record)
        except ValueError as exc:
            print(f"跳过 {record.run_id}：{exc}")
            continue
        prefix = (
            record.algorithm
            if vary is None
            else f"{vary}={required_argument(record, vary)}"
        )
        label = f"{prefix} ({record.run_id})"
        curves.append(Curve(label, record, points))
        rows.append(
            [
                label,
                f"{record.max_accuracy:.2f}%",
                f"{record.latest_accuracy:.2f}%",
                len(points),
            ]
        )
    if not curves:
        raise ValueError("没有可用于比较的有效准确率曲线")
    print_table(["运行", "最佳 acc", "最终 acc", "记录轮数"], rows)
    if explicit:
        print_baseline_differences(records)
    print("\n准确率曲线：")
    print(render_accuracy_chart(curves))


def aggregate_runs(grouped: dict[str, list[RunRecord]]) -> None:
    """输出固定 seed 下多次运行的终端统计与平均准确率曲线。"""
    curves: list[Curve] = []
    rows: list[list[Any]] = []
    for algorithm, records in grouped.items():
        latest = [record.latest_accuracy for record in records]
        maximum = [record.max_accuracy for record in records]
        trials = ",".join(str(trial_number(record)) for record in records)
        rows.append(
            [
                algorithm,
                trials,
                f"{mean(latest):.2f}% ± {stdev(latest):.2f}%",
                f"{mean(maximum):.2f}% ± {stdev(maximum):.2f}%",
            ]
        )
        rounds, averages, _ = aggregate_series(
            records, "accuracy", alignment="intersection"
        )
        if not rounds:
            raise ValueError(f"{algorithm} 指定试次没有共同的有效轮次")
        points = [
            (int(round_index), average)
            for round_index, average in zip(rounds, averages)
        ]
        curves.append(Curve(f"{algorithm} 平均", records[0], points))
    print_table(["算法", "试次", "最终 acc", "最佳 acc"], rows)
    print("\n平均准确率曲线：")
    print(render_accuracy_chart(curves))


def render_accuracy_chart(
    curves: list[Curve], width: int | None = None, height: int = 12
) -> str:
    """渲染无依赖的多条准确率 Unicode 散点折线预览。"""
    if not curves:
        return "没有曲线。"
    width = width or max(30, min(90, shutil.get_terminal_size((100, 24)).columns - 16))
    points = [point for curve in curves for point in curve.points]
    x_min, x_max = min(point[0] for point in points), max(point[0] for point in points)
    y_min, y_max = min(point[1] for point in points), max(point[1] for point in points)
    padding = max(0.5, (y_max - y_min) * 0.1)
    y_min, y_max = max(0.0, y_min - padding), min(100.0, y_max + padding)
    if y_max - y_min < 1.0:
        y_min, y_max = max(0.0, y_min - 0.5), min(100.0, y_max + 0.5)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for index, curve in enumerate(curves):
        symbol = SYMBOLS[index % len(SYMBOLS)]
        for round_index, accuracy in curve.points:
            column = (
                0
                if x_max == x_min
                else round((round_index - x_min) * (width - 1) / (x_max - x_min))
            )
            row = round((y_max - accuracy) * (height - 1) / (y_max - y_min))
            grid[row][column] = symbol if grid[row][column] == " " else "*"
    lines = []
    for row, values in enumerate(grid):
        value = y_max - row * (y_max - y_min) / (height - 1)
        lines.append(f"{value:6.2f} │" + "".join(values))
    lines.append("       └" + "─" * width)
    lines.append(
        f"        r{x_min}"
        + " " * max(1, width - len(str(x_min)) - len(str(x_max)) - 1)
        + f"r{x_max}"
    )
    lines.append(
        "图例："
        + "  ".join(
            f"{SYMBOLS[index % len(SYMBOLS)]} {curve.label}"
            for index, curve in enumerate(curves)
        )
    )
    return "\n".join(lines)


def run_results() -> int:
    """执行列表、单次查看或比较命令。"""
    args = parse_args()
    try:
        algorithms = parse_csv(args.algo)
        requested_trials = (
            [args.trial] if args.trial is not None else parse_trials(args.trials)
        )
        records = load_runs(args.database, algorithms or None, args.status)
        if requested_trials is not None:
            records = filter_runs(records, trial=requested_trials)
        if args.command == "list":
            list_runs(records, args.limit)
            return 0
        if args.command == "show":
            selected = next(
                (record for record in records if record.run_id == args.run_id), None
            )
            if selected is None:
                records = load_runs(args.database, status="all")
                selected = next(
                    (record for record in records if record.run_id == args.run_id), None
                )
            if selected is None:
                raise ValueError(f"找不到运行：{args.run_id}")
            print_run(selected)
            return 0
        if args.command == "aggregate":
            selected = select_trials_for_aggregate(
                records, algorithms, requested_trials
            )
            aggregate_runs(selected)
            return 0
        if args.run:
            requested = parse_csv(args.run)
            all_records = load_runs(args.database, status="all")
            indexed = {record.run_id: record for record in all_records}
            missing = [run_id for run_id in requested if run_id not in indexed]
            if missing:
                raise ValueError("找不到运行：" + ", ".join(missing))
            compare_runs([indexed[run_id] for run_id in requested], explicit=True)
            return 0
        if args.vary:
            if len(algorithms) != 1:
                raise ValueError("--vary 需要且只能指定一个 --algo")
            selected = select_parameter_comparison(records, algorithms[0], args.vary)
            compare_runs(selected, args.vary)
            return 0
        selected = select_latest_comparable(records, algorithms)
        compare_runs(selected)
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
