from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

RUNTIME_ARGUMENTS = {
    "algo",
    "data_root",
    "devices",
    "service_device",
    "log_level",
    "result_root",
    "times",
    "trials",
    "trial",
}
RUNTIME_FIELDS = RUNTIME_ARGUMENTS

COMMON_ARGUMENT_DEFAULTS: dict[str, Any] = {
    "dataset": "cifar10",
    "model": "cnn",
    "num_rounds": 100,
    "num_epochs": 10,
    "lr": 0.01,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "batch_size": 64,
    "num_clients": 20,
    "join_ratio": 1.0,
    "devices": "0:1",
    "service_device": 0,
    "seed": 42,
    "times": 1,
    "trials": None,
    "trial": 1,
    "test_ratio": 0.2,
    "partition": "dirichlet",
    "dir": 0.1,
    "classes_per_client": None,
    "feature_dim": 512,
    "pre_train": None,
    "check_round": 50,
    "data_root": "datasets",
    "result_root": "results",
    "adj_type": "ring",
    "edge_p": 0.2,
    "k_small_world": 4,
    "m_scale_free": 2,
    "log_level": "INFO",
}
BASELINE_FIELDS = (
    "dataset",
    "model",
    "partition",
    "dir",
    "classes_per_client",
    "num_clients",
    "join_ratio",
    "num_rounds",
    "num_epochs",
    "batch_size",
    "lr",
    "momentum",
    "weight_decay",
    "feature_dim",
    "test_ratio",
    "seed",
    "pre_train",
    "adj_type",
    "edge_p",
    "k_small_world",
    "m_scale_free",
)
DEFAULT_METRIC_COLUMNS = {"round", "train_loss", "accuracy", "elapsed_seconds"}
DATABASE_COLUMNS = {
    "run_id",
    "algorithm",
    "status",
    "started_at",
    "ended_at",
    "max_accuracy",
    "latest_accuracy",
    "run_path",
}


@dataclass
class Run:
    """一次运行的索引信息及按需加载的指标。"""

    run_id: str
    algorithm: str
    status: str
    started_at: str
    ended_at: str
    run_path: Path
    args: dict[str, Any] | None
    max_accuracy: float | None = None
    latest_accuracy: float | None = None
    issue: str | None = None
    metrics: list[dict[str, Any]] | None = None
    metric_columns: tuple[str, ...] = ()
    checkpoints: tuple[Path, ...] = field(default_factory=tuple)
    log_path: Path | None = None

    @property
    def path(self) -> Path:
        """Notebook API 的路径别名。"""
        return self.run_path

    @property
    def trial(self) -> int | None:
        return trial_number(self)

    def series(self, metric: str, x: str = "round") -> tuple[list[float], list[float]]:
        """提取逐轮指标，并支持累计耗时派生曲线。"""
        if metric == "cumulative_elapsed_seconds":
            xs, elapsed = self.series("elapsed_seconds", x)
            cumulative: list[float] = []
            total = 0.0
            for value in elapsed:
                total += value
                cumulative.append(total)
            return xs, cumulative
        rows, columns = (
            load_metric_rows(self)
            if self.metrics is None
            else (
                self.metrics,
                self.metric_columns,
            )
        )
        if x not in columns:
            raise KeyError(f"{self.run_id} 的 metrics.csv 不包含横坐标列 {x!r}。")
        if metric not in columns:
            raise KeyError(f"{self.run_id} 的 metrics.csv 不包含指标列 {metric!r}。")
        points = [
            (row.get(x), row.get(metric))
            for row in rows
            if is_number(row.get(x)) and is_number(row.get(metric))
        ]
        if not points:
            raise ValueError(f"{self.run_id} 的指标 {metric!r} 没有有效数值。")
        return [float(point[0]) for point in points], [
            float(point[1]) for point in points
        ]


RunRecord = Run


@dataclass(frozen=True)
class MetricTable:
    rows: list[dict[str, Any]]
    columns: tuple[str, ...]


def find_project_root(start: Path | None = None) -> Path:
    """从当前目录向上定位项目根目录。"""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "algo").is_dir():
            return candidate
    raise FileNotFoundError("无法定位 FL_MP 项目根目录。")


def resolve_run_path(raw_path: str | Path, root: Path) -> Path:
    """将 SQLite 中的绝对或相对运行路径统一解析。"""
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_args(run_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """读取参数对象；返回问题文本，供 CLI 容错展示或 notebook 严格报错。"""
    path = run_path / "args.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "缺少 args.json"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"args.json 无法读取：{exc}"
    if not isinstance(value, dict):
        return None, "args.json 顶层必须是对象"
    return value, None


def load_runs(
    database: Path | None = None,
    *,
    project_root: Path | None = None,
    algorithms: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    load_metrics: bool = False,
    required_metric_columns: Iterable[str] = (),
    strict: bool = False,
    require_artifacts: bool = False,
    exclude_test: bool = False,
) -> list[Run]:
    """从 SQLite 加载运行；支持按需读指标，并由调用方选择严格程度。"""
    root = (project_root or find_project_root()).resolve()
    database = database or root / "results" / "results.sqlite"
    if not database.is_file():
        raise FileNotFoundError(f"SQLite 结果库不存在：{database}")
    requested_algorithms = (
        None if algorithms is None else {name.lower() for name in algorithms}
    )
    requested_statuses = None if statuses is None else set(statuses)
    requested_ids = None if run_ids is None else set(run_ids)
    if (
        requested_algorithms == set()
        or requested_statuses == set()
        or requested_ids == set()
    ):
        raise ValueError("algorithms、statuses、run_ids 指定时均不得为空集合。")
    try:
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
            missing_columns = DATABASE_COLUMNS - columns
            if missing_columns:
                raise ValueError(
                    f"SQLite 结果库 {database} 缺少字段："
                    f"{', '.join(sorted(missing_columns))}"
                )
            id_rows = connection.execute("SELECT run_id FROM runs").fetchall()
            conditions = []
            if exclude_test:
                conditions.append("status != 'test'")
            parameters: list[str] = []
            if requested_algorithms is not None:
                names = sorted(requested_algorithms)
                conditions.append(
                    "lower(algorithm) IN (" + ", ".join("?" for _ in names) + ")"
                )
                parameters.extend(names)
            if requested_statuses is not None:
                values = sorted(requested_statuses)
                conditions.append("status IN (" + ", ".join("?" for _ in values) + ")")
                parameters.extend(values)
            if requested_ids is not None:
                ids = sorted(requested_ids)
                conditions.append("run_id IN (" + ", ".join("?" for _ in ids) + ")")
                parameters.extend(ids)
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            rows = connection.execute(
                f"SELECT * FROM runs{where} ORDER BY started_at DESC, run_id DESC",
                parameters,
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise ValueError(f"SQLite 结果库无法读取：{database}：{exc}") from exc
    known_ids = {str(row[0]) for row in id_rows}
    if requested_ids is not None and requested_ids - known_ids:
        raise ValueError(
            "SQLite 中找不到 run_id：" + ", ".join(sorted(requested_ids - known_ids))
        )

    runs: list[Run] = []
    for row in rows:
        run_id = row["run_id"]
        algorithm = row["algorithm"]
        status = row["status"]
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"SQLite 记录的 run_id 无效：{run_id!r}")
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise ValueError(f"SQLite 记录 {run_id} 的 algorithm 无效。")
        raw_path = row["run_path"]
        if not isinstance(raw_path, str) or not raw_path.strip():
            run_path = root
            issue = "SQLite 记录缺少有效 run_path"
        else:
            run_path = resolve_run_path(raw_path, root)
            issue = None
        args, args_issue = (
            read_args(run_path) if run_path.is_dir() else (None, "运行目录不存在")
        )
        issue = _join_issues(issue, args_issue)
        if exclude_test and args is not None and args.get("test_mode", False):
            continue
        args_algorithm = None if args is None else args.get("algo")
        if args is not None and not isinstance(args_algorithm, str):
            issue = _join_issues(issue, "args.json 缺少非空字符串 algo")
        elif (
            isinstance(args_algorithm, str)
            and args_algorithm.lower() != str(algorithm).lower()
        ):
            issue = _join_issues(issue, "SQLite algorithm 与 args.json 中 algo 不一致")
        run = Run(
            run_id=run_id,
            algorithm=str(algorithm).lower(),
            status=str(status),
            started_at=str(row["started_at"] or ""),
            ended_at=str(row["ended_at"] or ""),
            run_path=run_path,
            args=args,
            max_accuracy=_optional_float(row["max_accuracy"]),
            latest_accuracy=_optional_float(row["latest_accuracy"]),
            issue=issue,
        )
        if strict and issue:
            raise ValueError(f"运行 {run_id}：{issue}")
        if load_metrics:
            try:
                table = read_metrics_csv(
                    run_path / "metrics.csv",
                    required_columns=required_metric_columns,
                    contiguous_rounds=True,
                )
                run.metrics, run.metric_columns = table.rows, table.columns
            except (OSError, ValueError) as exc:
                if strict:
                    raise ValueError(f"运行 {run_id}：{exc}") from exc
                run.issue = _join_issues(run.issue, str(exc))
        if require_artifacts:
            log_path = run_path / "run.log"
            checkpoint_dir = run_path / "checkpoints"
            missing = []
            if not log_path.is_file():
                missing.append(f"运行日志不存在：{log_path}")
            if not checkpoint_dir.is_dir():
                missing.append(f"checkpoint 目录不存在：{checkpoint_dir}")
            if missing and strict:
                raise FileNotFoundError("；".join(missing))
            run.issue = _join_issues(run.issue, "；".join(missing) or None)
            run.log_path = log_path if log_path.is_file() else None
            run.checkpoints = (
                tuple(sorted(checkpoint_dir.glob("*.pt")))
                if checkpoint_dir.is_dir()
                else ()
            )
        runs.append(run)
    runs = sort_runs(runs)
    return runs


def read_metrics_csv(
    path: Path,
    *,
    required_columns: Iterable[str] = ("round", "accuracy"),
    contiguous_rounds: bool = False,
) -> MetricTable:
    """统一解析并校验指标 CSV；额外数值列会原样保留。"""
    if not path.is_file():
        raise FileNotFoundError(f"指标文件不存在：{path}")
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            columns = tuple(reader.fieldnames or ())
            if not columns:
                raise ValueError(f"指标文件没有表头：{path}")
            if len(columns) != len(set(columns)):
                raise ValueError(f"指标文件存在重复列名：{path}")
            missing = set(required_columns) - set(columns)
            if missing:
                raise ValueError(
                    f"指标文件 {path} 缺少必需列：{', '.join(sorted(missing))}"
                )
            rows = [
                {name: _convert_cell(value) for name, value in row.items()}
                for row in reader
            ]
    except (OSError, csv.Error, UnicodeError) as exc:
        raise ValueError(f"无法读取指标文件 {path}：{exc}") from exc
    if not rows:
        raise ValueError(f"指标文件没有数据行：{path}")
    seen_rounds: set[int] = set()
    for line_number, row in enumerate(rows, start=2):
        for name in columns:
            if not is_number(row.get(name)):
                raise ValueError(
                    f"指标文件 {path} 第 {line_number} 行的 {name} 不是有限数值："
                    f"{row.get(name)!r}"
                )
        round_index = row.get("round")
        if not isinstance(round_index, int) or round_index < 1:
            raise ValueError(
                f"指标文件 {path} 第 {line_number} 行 round 必须为正整数。"
            )
        if round_index in seen_rounds:
            raise ValueError(f"指标文件 {path} 存在重复轮次：{round_index}")
        seen_rounds.add(round_index)
        if contiguous_rounds and round_index != line_number - 1:
            raise ValueError(
                f"指标文件 {path} 第 {line_number} 行轮次不连续："
                f"期望 {line_number - 1}，实际 {round_index}。"
            )
        if "accuracy" in row and not 0 <= row["accuracy"] <= 100:
            raise ValueError(
                f"指标文件 {path} 第 {line_number} 行 accuracy 超出 [0, 100]。"
            )
        if "elapsed_seconds" in row and row["elapsed_seconds"] < 0:
            raise ValueError(
                f"指标文件 {path} 第 {line_number} 行 elapsed_seconds 不能为负数。"
            )
    return MetricTable(rows, columns)


def load_metric_rows(
    run: Run, required_columns: Iterable[str] = ("round", "accuracy")
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """返回运行的指标行；未预载时按需读取。"""
    if run.metrics is not None:
        missing = set(required_columns) - set(run.metric_columns)
        if missing:
            raise ValueError(f"{run.run_id} 缺少指标列：{', '.join(sorted(missing))}")
        return run.metrics, run.metric_columns
    table = read_metrics_csv(
        run.run_path / "metrics.csv", required_columns=required_columns
    )
    return table.rows, table.columns


def read_accuracy_points(run: Run) -> list[tuple[int, float]]:
    """读取 CLI 曲线所需的准确率点。"""
    try:
        rows, _ = load_metric_rows(run)
    except OSError as exc:
        raise ValueError(f"{run.run_id} 缺少或无法读取 metrics.csv：{exc}") from exc
    return sorted((int(row["round"]), float(row["accuracy"])) for row in rows)


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _convert_cell(value: str | None) -> Any:
    if value is None or value.strip() == "":
        return None
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        try:
            number = float(text)
            return number if math.isfinite(number) else text
        except ValueError:
            return text


def normalized_arguments(run: Run) -> dict[str, Any] | None:
    if run.args is None:
        return None
    return {**COMMON_ARGUMENT_DEFAULTS, **run.args}


def trial_number(run: Run) -> int | None:
    args = normalized_arguments(run)
    if args is None:
        return None
    value = args.get("trial", 1)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def argument(run: Run, name: str) -> Any:
    return "-" if run.args is None else run.args.get(name, "-")


def required_argument(run: Run, name: str) -> Any:
    if run.args is None or name not in run.args:
        raise ValueError(f"{run.run_id} 缺少参数 {name}")
    return run.args[name]


def filter_runs(runs: Sequence[Run], **filters: Any) -> list[Run]:
    """按索引字段或任意实验参数筛选运行；无匹配时返回空列表。"""
    selected = []
    for run in runs:
        args = normalized_arguments(run) or {}
        values = {
            "algorithm": run.algorithm,
            "algo": run.algorithm,
            "status": run.status,
            "run_id": run.run_id,
            "trial": trial_number(run),
            **args,
        }
        if all(
            _matches(values.get(name), expected) for name, expected in filters.items()
        ):
            selected.append(run)
    return selected


def sort_runs(
    runs: Iterable[Run], *, by: str = "started_at", reverse: bool = True
) -> list[Run]:
    if by not in {"started_at", "ended_at", "run_id", "algorithm"}:
        raise ValueError(f"不支持的运行排序字段：{by}")
    return sorted(runs, key=lambda run: (getattr(run, by), run.run_id), reverse=reverse)


def differing_arguments(
    runs: Sequence[Run], ignore: Iterable[str] = RUNTIME_ARGUMENTS
) -> dict[str, list[Any]]:
    ignored = set(ignore)
    names = set().union(*(run.args or {} for run in runs)) if runs else set()
    differences: dict[str, list[Any]] = {}
    for name in sorted(names - ignored):
        encoded = {
            json.dumps((run.args or {}).get(name), sort_keys=True, ensure_ascii=False)
            for run in runs
        }
        if len(encoded) > 1:
            differences[name] = [json.loads(value) for value in sorted(encoded)]
    return differences


def baseline_differences(runs: Sequence[Run]) -> list[str]:
    """返回基础实验控制变量中存在差异的字段。"""
    differences = []
    for name in BASELINE_FIELDS:
        values = {
            json.dumps(args[name], sort_keys=True)
            for run in runs
            if (args := normalized_arguments(run)) is not None
        }
        if len(values) > 1:
            differences.append(name)
    return differences


def reject_duplicate_runs(
    runs: Sequence[Run], ignored: Iterable[str] = RUNTIME_ARGUMENTS
) -> None:
    grouped: dict[tuple[str, int | None], list[str]] = defaultdict(list)
    ignored_names = set(ignored)
    for run in runs:
        arguments = {
            name: value
            for name, value in (run.args or {}).items()
            if name not in ignored_names
        }
        signature = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        grouped[(signature, trial_number(run))].append(run.run_id)
    duplicates = [run_ids for run_ids in grouped.values() if len(run_ids) > 1]
    if duplicates:
        detail = "；".join(", ".join(run_ids) for run_ids in duplicates)
        raise ValueError("存在相同配置和 trial 的重复运行，请显式筛选：" + detail)


def baseline_signature(run: Run) -> tuple[tuple[str, str], ...] | None:
    args = normalized_arguments(run)
    if args is None:
        return None
    return tuple(
        (name, json.dumps(args[name], sort_keys=True, ensure_ascii=False))
        for name in BASELINE_FIELDS
    )


def baseline_description(run: Run) -> str:
    return ", ".join(
        f"{name}={argument(run, name)}"
        for name in ("dataset", "partition", "num_clients", "num_rounds", "seed")
    )


def parameter_signature(run: Run, vary: str) -> tuple[tuple[str, str], ...]:
    args = normalized_arguments(run)
    if args is None:
        return ()
    return tuple(
        (name, json.dumps(value, sort_keys=True, ensure_ascii=False))
        for name, value in sorted(args.items())
        if name not in RUNTIME_ARGUMENTS and name != vary
    )


def select_latest_comparable(
    runs: Sequence[Run], algorithms: Sequence[str]
) -> list[Run]:
    requested = set(algorithms)
    grouped: dict[tuple[tuple[str, str], ...], dict[str, list[Run]]] = {}
    for run in runs:
        if run.algorithm not in requested or run.status != "completed":
            continue
        signature = baseline_signature(run)
        if signature is not None:
            grouped.setdefault(signature, {}).setdefault(run.algorithm, []).append(run)
    candidates = [
        by_algorithm
        for by_algorithm in grouped.values()
        if requested <= set(by_algorithm)
    ]
    if not candidates:
        descriptions = list(
            dict.fromkeys(
                baseline_description(run)
                for run in runs
                if run.algorithm in requested and run.status == "completed"
            )
        )
        detail = "。可用基础配置：" + "；".join(descriptions) if descriptions else ""
        raise ValueError("找不到同时包含指定算法的可比完成运行" + detail)
    selected_group = max(
        candidates,
        key=lambda group: max(
            run.started_at for group_runs in group.values() for run in group_runs
        ),
    )
    return [
        max(selected_group[algorithm], key=lambda run: run.started_at)
        for algorithm in algorithms
    ]


def parameter_groups(
    runs: Sequence[Run], algorithm: str, vary: str | Sequence[str]
) -> dict[tuple[Any, ...], list[Run]]:
    """按一项或多项算法参数分组，供终端和 notebook 共用。"""
    names = [vary] if isinstance(vary, str) else list(vary)
    if not names:
        raise ValueError("vary 至少需要一个参数名。")
    groups: dict[tuple[Any, ...], list[Run]] = defaultdict(list)
    for run in runs:
        if run.algorithm != algorithm or run.args is None:
            continue
        if all(name in run.args for name in names):
            groups[tuple(run.args[name] for name in names)].append(run)
    return dict(groups)


def select_parameter_comparison(
    runs: Sequence[Run], algorithm: str, vary: str
) -> list[Run]:
    candidates: dict[tuple[tuple[str, str], ...], dict[str, list[Run]]] = {}
    for run in runs:
        if run.algorithm != algorithm or run.status != "completed" or run.args is None:
            continue
        if vary not in run.args:
            continue
        values = candidates.setdefault(parameter_signature(run, vary), {})
        key = json.dumps(required_argument(run, vary), sort_keys=True)
        values.setdefault(key, []).append(run)
    candidates = {
        signature: values
        for signature, values in candidates.items()
        if len(values) >= 2
    }
    if not candidates:
        raise ValueError(f"找不到 {algorithm} 中仅 {vary} 不同的至少两条完成运行")
    selected = max(
        candidates.values(),
        key=lambda values: max(
            run.started_at for group in values.values() for run in group
        ),
    )
    return sorted(
        [max(group, key=lambda run: run.started_at) for group in selected.values()],
        key=lambda run: json.dumps(required_argument(run, vary), sort_keys=True),
    )


def select_trials_for_aggregate(
    runs: Sequence[Run], algorithms: Sequence[str], trials: Sequence[int] | None
) -> dict[str, list[Run]]:
    requested = set(algorithms)
    grouped: dict[tuple[tuple[str, str], ...], dict[str, dict[int, list[Run]]]] = {}
    for run in runs:
        trial = trial_number(run)
        signature = baseline_signature(run)
        if (
            run.status == "completed"
            and run.algorithm in requested
            and trial is not None
            and signature is not None
        ):
            grouped.setdefault(signature, {}).setdefault(run.algorithm, {}).setdefault(
                trial, []
            ).append(run)
    candidates = []
    for by_algorithm in grouped.values():
        if not requested <= set(by_algorithm):
            continue
        available = set.intersection(
            *(set(by_algorithm[algorithm]) for algorithm in algorithms)
        )
        selected_trials = list(trials) if trials is not None else sorted(available)
        if len(selected_trials) >= 2 and set(selected_trials) <= available:
            candidates.append((by_algorithm, selected_trials))
    if not candidates:
        requested_text = (
            "全部共有试次" if trials is None else ",".join(map(str, trials))
        )
        raise ValueError(
            f"找不到覆盖试次 {requested_text} 的同配置完成运行（聚合至少需要两次）"
        )
    by_algorithm, selected_trials = max(
        candidates,
        key=lambda item: max(
            run.started_at
            for algorithm in algorithms
            for trial in item[1]
            for run in item[0][algorithm][trial]
        ),
    )
    return {
        algorithm: [
            max(by_algorithm[algorithm][trial], key=lambda run: run.started_at)
            for trial in selected_trials
        ]
        for algorithm in algorithms
    }


def aggregate_series(
    runs: Sequence[Run],
    metric: str,
    x: str = "round",
    *,
    alignment: str = "union",
) -> tuple[list[float], list[float], list[float]]:
    """按轮次计算均值和样本标准差；可取轮次并集或交集。"""
    if alignment not in {"union", "intersection"}:
        raise ValueError("alignment 必须为 'union' 或 'intersection'")
    values_by_run = []
    for run in runs:
        xs, ys = run.series(metric, x)
        values_by_run.append(dict(zip(xs, ys)))
    if not values_by_run:
        raise ValueError("没有可聚合的运行。")
    coordinates = (
        set.union(*(set(values) for values in values_by_run))
        if alignment == "union"
        else set.intersection(*(set(values) for values in values_by_run))
    )
    coordinates = sorted(coordinates)
    means = []
    deviations = []
    for coordinate in coordinates:
        values = [
            series[coordinate] for series in values_by_run if coordinate in series
        ]
        means.append(fmean(values))
        deviations.append(stdev(values) if len(values) > 1 else 0.0)
    return coordinates, means, deviations


def get_run(run_or_id: Run | str, runs: Sequence[Run]) -> Run:
    if isinstance(run_or_id, Run):
        return run_or_id
    matches = [run for run in runs if run.run_id == run_or_id]
    if len(matches) != 1:
        raise ValueError(f"运行 {run_or_id!r} 匹配数量为 {len(matches)}。")
    return matches[0]


def read_log_text(run: Run, tail: int | None = 100) -> str:
    """读取日志全文或末尾若干行。"""
    path = run.log_path or run.run_path / "run.log"
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines if tail is None else lines[-tail:])


def checkpoint_path(run: Run, round_index: int | None = None) -> Path:
    """解析指定或最新 checkpoint，并统一校验命名和轮次。"""
    directory = run.run_path / "checkpoints"
    if round_index is not None:
        if round_index < 1:
            raise ValueError("checkpoint 轮次必须为正整数")
        path = directory / f"round_{round_index}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    checkpoints = run.checkpoints or tuple(directory.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"{run.run_id} 没有 checkpoint。")
    candidates = []
    for path in checkpoints:
        try:
            checkpoint_round = int(path.stem.rsplit("_", 1)[-1])
        except ValueError as exc:
            raise ValueError(f"checkpoint 文件名不符合 round_N.pt：{path}") from exc
        if checkpoint_round < 1:
            raise ValueError(f"checkpoint 轮次必须为正整数：{path}")
        candidates.append((checkpoint_round, path))
    return max(candidates)[1]


def _matches(actual: Any, expected: Any) -> bool:
    if callable(expected):
        return bool(expected(actual))
    if isinstance(expected, (list, tuple, set, frozenset)):
        return actual in expected
    return actual == expected


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _join_issues(*issues: str | None) -> str | None:
    present = [issue for issue in issues if issue]
    return "；".join(present) if present else None
