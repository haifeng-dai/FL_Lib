import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, TypeGuard

import matplotlib.pyplot as plt

from core.result import RunSummary


# 论文常用配色方案
COLOR_PALETTES: dict[str, list[str]] = {
    # Nature 2023 (Burke et al.) 论文配色：前三色为折线主色（灰蓝、桃橙、珊瑚粉）
    "nature": [
        "#92B4C8",  # 板岩灰蓝 / Slate Blue
        "#FAC795",  # 浅桃橙 / Peach
        "#EEA599",  # 珊瑚粉 / Coral
        "#ABD3E1",  # 浅天蓝 / Light Blue
        "#E3EDE0",  # 鼠尾草浅绿 / Sage
        "#FFE9BE",  # 浅米黄 / Cream
    ],
    # Cell 2023 (Wang et al.) 论文折线图三色（天空蓝、珊瑚红、丁香紫）
    "cell_line": [
        "#7DAEE0",  # 天空蓝 / Sky Blue
        "#EA8379",  # 珊瑚红 / Coral Red
        "#B395BD",  # 丁香紫 / Lilac Purple
    ],
    # Cell 2023 (Wang et al.) 论文柱状/比例图三色（墨松绿、暖沙黄、陶土红）
    "cell_bar": [
        "#299D8F",  # 墨松绿 / Teal
        "#E9C46A",  # 暖沙黄 / Sand Yellow
        "#D87659",  # 陶土红 / Terracotta
    ],
    "line": [
        "#d62728",  # brick red
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
    ],
}


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def algorithm_names(directory: Path) -> set[str]:
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


def metric_series(
    run: RunSummary,
    metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    metric: str,
    x: str = "round",
) -> tuple[list[float], list[float]]:
    rows = metrics.get(run.run_id, ())
    if metric == "cumulative_elapsed_seconds":
        xs, elapsed = metric_series(run, metrics, "elapsed_seconds", x)
        total = 0.0
        cumulative = []
        for value in elapsed:
            total += value
            cumulative.append(total)
        return xs, cumulative
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        xv = row.get(x)
        yv = row.get(metric)
        if _is_number(xv) and _is_number(yv):
            xs.append(float(xv))
            ys.append(float(yv))
    if not xs:
        raise ValueError(f"{run.run_id} 没有可用的 {metric!r} 指标。")
    return xs, ys


def runs_table(
    runs: Sequence[RunSummary],
    metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    limit: int | None = 30,
) -> list[dict[str, Any]]:
    shown = runs if limit is None else runs[:limit]
    return [
        {
            "run_id": run.run_id,
            "algorithm": run.algorithm,
            "status": run.status,
            "dataset": run.args.get("dataset"),
            "partition": run.args.get("partition"),
            "topology": run.args.get("adj_type"),
            "trial": run.args.get("trial"),
            "rounds": len(metrics.get(run.run_id, ())),
            "best_acc": run.max_accuracy,
            "final_acc": run.latest_accuracy,
            "checkpoints": len(
                tuple((run.run_path / "checkpoints").glob("round_*.pt"))
            ),
        }
        for run in shown
    ]


def plot_run(
    run: RunSummary,
    metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_metrics: Sequence[str] | None = None,
    x: str = "round",
):
    rows = metrics.get(run.run_id, ())
    available = list(dict.fromkeys(name for row in rows for name in row))
    names = list(
        selected_metrics
        or [
            name
            for name in available
            if name != x and any(_is_number(row.get(name)) for row in rows)
        ]
    )
    if "elapsed_seconds" in names:
        names.append("cumulative_elapsed_seconds")
    series = [(name, *metric_series(run, metrics, name, x)) for name in names]
    columns = min(2, len(series))
    if columns == 0:
        raise ValueError(f"{run.run_id} 没有可绘制指标。")
    fig, axes = plt.subplots(
        math.ceil(len(series) / columns), columns,
        figsize=(6.2 * columns, 3.6 * math.ceil(len(series) / columns)),
        squeeze=False,
    )
    for axis, (name, xs, ys) in zip(axes.flat, series):
        axis.plot(xs, ys, linewidth=1.8)
        axis.set(title=name, xlabel=x, ylabel=name)
    for axis in list(axes.flat)[len(series):]:
        axis.set_visible(False)
    fig.suptitle(f"{run.algorithm} · {run.run_id}", y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_algorithm_runs(
    runs: Sequence[RunSummary],
    metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    algorithm: str,
    metric: str = "accuracy",
):
    selected = [run for run in runs if run.algorithm == algorithm]
    fig, axis = plt.subplots(figsize=(9, 5))
    for run in selected:
        xs, ys = metric_series(run, metrics, metric)
        axis.plot(xs, ys, alpha=0.8,
                  label=f"{run.run_id} (trial={run.args.get('trial')})")
    if not axis.lines:
        plt.close(fig)
        raise ValueError(f"没有 {algorithm} 的 {metric} 指标。")
    axis.set(title=f"{algorithm} · {metric}", xlabel="round", ylabel=metric)
    axis.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, axis, selected


def plot_algorithm_comparison(
    runs: Sequence[RunSummary],
    metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    algorithms: Sequence[str],
    metric: str = "accuracy",
    show_std: bool = True,
):
    groups = {
        algorithm: [run for run in runs if run.algorithm == algorithm]
        for algorithm in algorithms
    }
    missing = [name for name, group in groups.items() if not group]
    if missing:
        raise ValueError("没有匹配运行的算法：" + ", ".join(missing))
    return (*_draw_groups(groups, metrics, metric, f"algorithm comparison · {metric}", show_std), runs)


def plot_parameter_comparison(
    runs: Sequence[RunSummary],
    metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    algorithm: str,
    vary: str | Sequence[str],
    metric: str = "accuracy",
    show_std: bool = True,
):
    names = [vary] if isinstance(vary, str) else list(vary)
    groups: dict[tuple[Any, ...], list[RunSummary]] = defaultdict(list)
    for run in runs:
        if run.algorithm == algorithm and all(name in run.args for name in names):
            groups[tuple(run.args[name] for name in names)].append(run)
    if len(groups) < 2:
        raise ValueError(f"{algorithm} 当前不足两个参数组合：{names}")
    labeled = {
        ", ".join(f"{name}={value}" for name, value in zip(names, key)): group
        for key, group in sorted(groups.items(), key=lambda item: str(item[0]))
    }
    return (*_draw_groups(labeled, metrics, metric, f"{algorithm} parameter comparison · {metric}", show_std), labeled)


def _draw_groups(groups, metrics, metric, title, show_std):
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for label, group in groups.items():
        curves = [dict(zip(*metric_series(run, metrics, metric)))
                  for run in group]
        xs = sorted(set().union(*(curve.keys() for curve in curves)))
        means, deviations = [], []
        for x in xs:
            values = [curve[x] for curve in curves if x in curve]
            means.append(fmean(values))
            deviations.append(stdev(values) if len(values) > 1 else 0.0)
        (line,) = axis.plot(xs, means, linewidth=2,
                            label=f"{label} (n={len(group)})")
        if show_std and len(group) > 1:
            axis.fill_between(
                xs,
                [mean - sd for mean, sd in zip(means, deviations)],
                [mean + sd for mean, sd in zip(means, deviations)],
                color=line.get_color(), alpha=0.15,
            )
    axis.set(title=title, xlabel="round", ylabel=metric)
    axis.legend()
    fig.tight_layout()
    return fig, axis


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
