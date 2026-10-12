import csv
import hashlib
import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

MetricColumn = str
MetricValue = int | float | None
MetricsRow = dict[str, MetricValue]

BASE_METRIC_COLUMNS = ("round", "train_loss", "accuracy", "elapsed_seconds")


@dataclass(frozen=True)
class RunPaths:
    """一次运行的文件路径。"""

    directory: Path
    log_file: Path
    metrics_file: Path
    checkpoint_dir: Path
    database: Path
    run_id: str
    metric_columns: tuple[str, ...]
    required_metric_columns: tuple[str, ...]


def create_run(
    args,
    *,
    client_metric: Literal["global_acc", "client_macro_acc"] = "client_macro_acc",
    extra_metric_columns: tuple[MetricColumn, ...] = (),
) -> RunPaths:
    """创建运行目录、写入参数并初始化日志和指标文件。"""
    arguments = vars(args).copy()
    encoded = json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode()
    digest = hashlib.blake2s(encoded, digest_size=3).hexdigest()
    now = datetime.now(UTC)
    run_id = f"{now:%H%M%S}-{args.algo.lower()}-trial-{args.trial:03d}-{digest}"
    directory = (
        Path(args.result_root) / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}" / run_id
    )
    required_columns = BASE_METRIC_COLUMNS
    if getattr(args, "eval_clients", False):
        required_columns += (client_metric,)
    columns = required_columns + extra_metric_columns
    if len(columns) != len(set(columns)):
        raise ValueError("metrics.csv 指标列名重复")

    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    (directory / "args.json").write_text(
        json.dumps(arguments, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths = RunPaths(
        directory=directory,
        log_file=directory / "run.log",
        metrics_file=directory / "metrics.csv",
        checkpoint_dir=checkpoints,
        database=Path(args.result_root) / "results.sqlite",
        run_id=run_id,
        metric_columns=columns,
        required_metric_columns=required_columns,
    )
    with paths.metrics_file.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=paths.metric_columns).writeheader()
    return paths


def configure_logging(paths: RunPaths, level: str) -> logging.Logger:
    """配置仅属于当前运行的日志记录器。"""
    logger = logging.getLogger(f"fl_mp.{paths.run_id}")
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.FileHandler(paths.log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s\n%(message)s"))
    logger.addHandler(handler)
    return logger


def append_metrics(paths: RunPaths, metrics: MetricsRow) -> None:
    """按运行创建时声明的指标 schema 追加一行。"""
    unknown = set(metrics) - set(paths.metric_columns)
    if unknown:
        raise ValueError("未声明的指标列：" + ", ".join(sorted(unknown)))
    missing = set(paths.required_metric_columns) - set(metrics)
    if missing:
        raise ValueError("缺少必需的指标：" + ", ".join(sorted(missing)))

    row: dict[str, Any] = {}
    for name in paths.metric_columns:
        value = metrics.get(name)
        if value is None:
            if name in paths.required_metric_columns:
                raise ValueError(f"必需指标 {name!r} 不能留空")
            row[name] = ""
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"指标 {name!r} 必须是数值或 None")
        if not math.isfinite(value):
            raise ValueError(f"指标 {name!r} 必须是有限数值")
        row[name] = value

    with paths.metrics_file.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=paths.metric_columns).writerow(row)


def ensure_runs_table(connection: sqlite3.Connection) -> None:
    """创建全局运行摘要表。"""
    connection.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, algorithm TEXT NOT NULL, started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL, status TEXT NOT NULL, max_accuracy REAL NOT NULL,
            latest_accuracy REAL NOT NULL, run_path TEXT NOT NULL, args_hash TEXT NOT NULL
        )"""
    )


def record_running(paths: RunPaths, args, started: datetime) -> None:
    """在训练开始前登记可查询的运行中状态。"""
    paths.database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(paths.database) as connection:
        ensure_runs_table(connection)
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paths.run_id,
                args.algo,
                started.isoformat(),
                "",
                "running",
                0.0,
                0.0,
                str(paths.directory),
                paths.run_id.rsplit("-", 1)[1],
            ),
        )


def record_summary(
    paths: RunPaths,
    args,
    started: datetime,
    ended: datetime,
    status: str,
    max_accuracy: float,
    latest_accuracy: float,
) -> None:
    """在全局 SQLite 摘要库登记本次运行。"""
    paths.database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(paths.database) as connection:
        ensure_runs_table(connection)
        connection.execute(
            """INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                ended_at = excluded.ended_at,
                status = excluded.status,
                max_accuracy = excluded.max_accuracy,
                latest_accuracy = excluded.latest_accuracy""",
            (
                paths.run_id,
                args.algo,
                started.isoformat(),
                ended.isoformat(),
                status,
                max_accuracy,
                latest_accuracy,
                str(paths.directory),
                paths.run_id.rsplit("-", 1)[1],
            ),
        )
