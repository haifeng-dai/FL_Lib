"""FL_MP 实验结果 notebook 的共享读取、筛选、聚合与绘图工具。"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.save.results import (
    DEFAULT_METRIC_COLUMNS as REQUIRED_METRIC_COLUMNS,
)
from core.save.results import (
    RUNTIME_ARGUMENTS,
    Run,
    aggregate_series,
    checkpoint_path,
    differing_arguments,
    filter_runs,
    find_project_root,
    get_run,
    is_number,
    load_runs,
    parameter_groups,
    read_log_text,
    reject_duplicate_runs,
)

__all__ = [
    "Run",
    "algorithm_names",
    "configure_plots",
    "discover_runs",
    "exclude_test_runs",
    "family_runs",
    "load_checkpoint",
    "numeric_metrics",
    "plot_algorithm_comparison",
    "plot_algorithm_runs",
    "plot_parameter_comparison",
    "plot_run",
    "require_plot_run",
    "read_log",
    "reject_duplicate_runs",
    "runs_table",
    "select_runs",
    "show_run_details",
]


def configure_plots() -> None:
    """设置适合 notebook 的统一 Matplotlib 风格。"""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def algorithm_names(directory: Path) -> set[str]:
    """从算法包自动发现算法名，新增算法文件后无需修改 notebook。"""
    if not directory.is_dir():
        raise FileNotFoundError(f"算法目录不存在：{directory}")
    names = {
        path.stem.lower()
        for path in directory.glob("*.py")
        if path.stem not in {"__init__", "test"}
    }
    if not names:
        raise ValueError(f"算法目录中没有可用算法文件：{directory}")
    return names


def discover_runs(
    project_root: Path | None = None,
    result_root: Path | None = None,
    database: Path | None = None,
    algorithms: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
) -> list[Run]:
    """读取严格校验的记录；默认排除测试运行，显式 statuses 可查看测试。"""
    root = (project_root or find_project_root()).resolve()
    if database is None and result_root is not None:
        database = result_root.resolve() / "results.sqlite"
    runs = load_runs(
        database,
        project_root=root,
        algorithms=algorithms,
        run_ids=run_ids,
        statuses=statuses,
        load_metrics=True,
        required_metric_columns=REQUIRED_METRIC_COLUMNS,
        strict=True,
        require_artifacts=True,
        exclude_test=statuses is None,
    )
    if not runs:
        raise FileNotFoundError("没有找到符合条件的运行。")
    return runs


def family_runs(runs: Sequence[Run], names: Iterable[str]) -> list[Run]:
    """按算法名集合筛选一个算法族。"""
    selected = filter_runs(runs, algorithm={name.lower() for name in names})
    if not selected:
        raise ValueError("没有找到该算法族的任何实验结果。")
    return selected


def select_runs(runs: Sequence[Run], **filters: Any) -> list[Run]:
    """按运行属性或任意实验参数筛选运行。"""
    selected = filter_runs(runs, **filters)
    if not selected:
        description = ", ".join(f"{name}={value!r}" for name, value in filters.items())
        raise ValueError(f"没有符合筛选条件的实验结果：{description or '无筛选条件'}")
    return selected


def runs_table(runs: Sequence[Run], limit: int | None = 30) -> list[dict[str, Any]]:
    """生成适合 notebook 直接显示的运行概览。"""
    if not runs:
        raise ValueError("没有可显示的实验结果。")
    shown = list(runs) if limit is None else list(runs)[:limit]
    return [
        {
            "run_id": run.run_id,
            "algorithm": run.algorithm,
            "status": run.status,
            "dataset": (run.args or {}).get("dataset"),
            "partition": (run.args or {}).get("partition"),
            "topology": (run.args or {}).get("adj_type"),
            "trial": run.trial,
            "rounds": len(run.metrics or []),
            "best_acc": run.max_accuracy,
            "final_acc": run.latest_accuracy,
            "checkpoints": len(run.checkpoints),
        }
        for run in shown
    ]


def numeric_metrics(run: Run, x: str = "round") -> list[str]:
    """列出一次运行的全部可绘制指标。"""
    metrics = [
        name
        for name in run.metric_columns
        if name != x and any(is_number(row.get(name)) for row in run.metrics)
    ]
    if "elapsed_seconds" in metrics:
        metrics.append("cumulative_elapsed_seconds")
    return metrics


def _is_test_run(run: Run) -> bool:
    """识别成功、运行中或失败的测试模式运行。"""
    return run.status == "test" or bool((run.args or {}).get("test_mode", False))


def exclude_test_runs(runs: Sequence[Run]) -> list[Run]:
    """返回可用于绘图的正式运行。"""
    return [run for run in runs if not _is_test_run(run)]


def require_plot_run(run: Run) -> None:
    """拒绝将单次测试运行传给绘图函数。"""
    if _is_test_run(run):
        raise ValueError(f"测试运行 {run.run_id} 不参与绘图。")


def plot_run(
    run_or_id: Run | str,
    runs: Sequence[Run],
    metrics: Sequence[str] | None = None,
    x: str = "round",
):
    """绘制一次运行的多个指标。"""
    run = get_run(run_or_id, runs)
    require_plot_run(run)
    selected_metrics = list(metrics or numeric_metrics(run, x))
    if not selected_metrics:
        raise ValueError(f"{run.run_id} 没有可绘制的数值指标。")
    series = [(metric, *run.series(metric, x)) for metric in selected_metrics]
    columns = min(2, len(selected_metrics))
    rows = math.ceil(len(selected_metrics) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6.2 * columns, 3.6 * rows), squeeze=False
    )
    for axis, (metric, xs, ys) in zip(axes.flat, series):
        axis.plot(xs, ys, linewidth=1.8)
        axis.set(title=metric, xlabel=x, ylabel=metric)
    for axis in list(axes.flat)[len(selected_metrics) :]:
        axis.set_visible(False)
    fig.suptitle(f"{run.algorithm} · {run.run_id}", y=1.01)
    fig.tight_layout()
    return fig, axes


def _draw_groups(groups: dict[str, list[Run]], metric: str, title: str, show_std: bool):
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for label, group in groups.items():
        xs, means, deviations = aggregate_series(group, metric)
        if not xs:
            continue
        (line,) = axis.plot(xs, means, linewidth=2, label=f"{label} (n={len(group)})")
        if show_std and len(group) > 1:
            axis.fill_between(
                xs,
                [mean - deviation for mean, deviation in zip(means, deviations)],
                [mean + deviation for mean, deviation in zip(means, deviations)],
                color=line.get_color(),
                alpha=0.15,
            )
    if not axis.lines:
        plt.close(fig)
        raise ValueError(f"匹配运行中没有指标 {metric!r}。")
    axis.set(title=title, xlabel="round", ylabel=metric)
    axis.legend()
    fig.tight_layout()
    return fig, axis


def plot_algorithm_runs(
    runs: Sequence[Run],
    algorithm: str,
    metric: str = "accuracy",
    *,
    filters: dict[str, Any] | None = None,
):
    """分别绘制一个算法的所有匹配运行。"""
    selected = select_runs(
        exclude_test_runs(runs), algorithm=algorithm, **(filters or {})
    )
    if not selected:
        raise ValueError(f"没有找到 {algorithm} 的匹配运行。")
    fig, axis = plt.subplots(figsize=(9, 5))
    for run in reversed(selected):
        xs, ys = run.series(metric)
        if xs:
            axis.plot(xs, ys, alpha=0.8, label=f"{run.run_id} (trial={run.trial})")
    if not axis.lines:
        plt.close(fig)
        raise ValueError(f"匹配运行中没有指标 {metric!r}。")
    axis.set(title=f"{algorithm} · {metric}", xlabel="round", ylabel=metric)
    axis.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, axis, selected


def plot_algorithm_comparison(
    runs: Sequence[Run],
    algorithms: Sequence[str],
    metric: str = "accuracy",
    *,
    filters: dict[str, Any] | None = None,
    show_std: bool = True,
):
    """比较多个算法；每个算法内部按 trial 聚合。"""
    selected = select_runs(
        exclude_test_runs(runs), algorithm=set(algorithms), **(filters or {})
    )
    groups = {
        algorithm: [run for run in selected if run.algorithm == algorithm]
        for algorithm in algorithms
    }
    missing = [algorithm for algorithm, group in groups.items() if not group]
    if missing:
        raise ValueError("没有匹配运行的算法：" + ", ".join(missing))
    fig, axis = _draw_groups(
        groups, metric, f"algorithm comparison · {metric}", show_std
    )
    differences = differing_arguments(selected)
    if differences:
        print("提醒：匹配运行中仍存在以下非运行时参数差异：")
        print(json.dumps(differences, ensure_ascii=False, indent=2))
    return fig, axis, selected


def plot_parameter_comparison(
    runs: Sequence[Run],
    algorithm: str,
    vary: str | Sequence[str],
    metric: str = "accuracy",
    *,
    filters: dict[str, Any] | None = None,
    show_std: bool = True,
):
    """比较单算法的一项或多项参数组合。"""
    names = [vary] if isinstance(vary, str) else list(vary)
    if not names:
        raise ValueError("vary 至少需要一个参数名。")
    selected = select_runs(
        exclude_test_runs(runs), algorithm=algorithm, **(filters or {})
    )
    raw_groups = parameter_groups(selected, algorithm, names)
    if len(raw_groups) < 2:
        raise ValueError(f"{algorithm} 在当前条件下不足两个参数组合：{names}")
    groups = {
        ", ".join(f"{name}={value}" for name, value in zip(names, key)): group
        for key, group in sorted(raw_groups.items(), key=lambda item: str(item[0]))
    }
    fig, axis = _draw_groups(
        groups, metric, f"{algorithm} parameter comparison · {metric}", show_std
    )
    remaining = differing_arguments(selected, RUNTIME_ARGUMENTS | set(names))
    if remaining:
        print("提醒：除对比参数外，以下参数仍有差异：")
        print(json.dumps(remaining, ensure_ascii=False, indent=2))
    return fig, axis, groups


def show_run_details(run_or_id: Run | str, runs: Sequence[Run]) -> dict[str, Any]:
    """显示一次运行的参数、指标列和原始文件位置。"""
    run = get_run(run_or_id, runs)
    return {
        "run_id": run.run_id,
        "algorithm": run.algorithm,
        "status": run.status,
        "path": str(run.path),
        "args": run.args,
        "metric_columns": run.metric_columns,
        "metric_rows": len(run.metrics),
        "checkpoints": [str(path) for path in run.checkpoints],
        "log_path": str(run.log_path) if run.log_path else None,
    }


def read_log(run_or_id: Run | str, runs: Sequence[Run], tail: int | None = 100) -> str:
    """读取完整日志或最后若干行。"""
    return read_log_text(get_run(run_or_id, runs), tail)


def load_checkpoint(
    run_or_id: Run | str,
    runs: Sequence[Run],
    round_index: int | None = None,
) -> dict[str, Any]:
    """按需将 checkpoint 安全映射到 CPU。"""
    import torch

    run = get_run(run_or_id, runs)
    return torch.load(
        checkpoint_path(run, round_index), map_location="cpu", weights_only=False
    )
