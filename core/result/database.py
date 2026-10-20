import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .defaults import defaults_for_algorithm


@dataclass(frozen=True)
class RunQuery:
    """运行筛选条件。

    不同条件按 AND 匹配。args 中的标量、列表和字典按精确值匹配；元组表示
    同一参数字段的多个候选值，按 OR 匹配。索引字段的元组值也按 OR 匹配。
    查询会自动合并所选算法家族的公共默认参数；传入的 args 会覆盖相同字段。
    None 表示不筛选；空元组表示无匹配结果。
    """

    run_ids: tuple[str, ...] | None = None
    algorithms: tuple[str, ...] | None = None
    statuses: tuple[str, ...] | None = None
    args: Mapping[str, Any] = field(default_factory=dict)
    apply_defaults: bool = True


@dataclass(frozen=True)
class RunSummary:
    """数据库中的运行索引和参数，不包含完整逐轮指标。"""

    run_id: str
    algorithm: str
    started_at: str
    ended_at: str
    status: str
    max_accuracy: float
    latest_accuracy: float
    run_path: Path
    args_hash: str
    args: dict[str, Any]
    latest_round: int


class DatabaseReader:
    """读取 SQLite 中的运行记录和逐轮指标。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def query_runs(self, query: RunQuery | None = None) -> list[RunSummary]:
        """按条件读取运行摘要和参数，默认按开始时间倒序。"""
        if not self.path.is_file():
            raise FileNotFoundError(f"结果数据库不存在：{self.path}")
        query = query or RunQuery(apply_defaults=False)
        run_ids = self._values(query.run_ids)
        algorithms = self._values(query.algorithms)
        statuses = self._values(query.statuses)

        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
            selected = [
                "run_id",
                "algorithm",
                "started_at",
                "ended_at",
                "status",
                "max_accuracy",
                "latest_accuracy",
                "run_path",
                "args_hash",
                "args_json" if "args_json" in columns else "'{}' AS args_json",
                "latest_round" if "latest_round" in columns else "0 AS latest_round",
            ]
            conditions: list[str] = []
            parameters: list[str] = []
            self._add_filter(conditions, parameters, "run_id", run_ids)
            self._add_filter(conditions, parameters, "lower(algorithm)", algorithms)
            self._add_filter(conditions, parameters, "status", statuses)
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            rows = connection.execute(
                f"SELECT {', '.join(selected)} FROM runs{where} "
                "ORDER BY started_at DESC, run_id DESC",
                parameters,
            ).fetchall()

        summaries = [self._summary(row) for row in rows]
        apply_defaults = query.apply_defaults and query.run_ids is None
        if apply_defaults or query.args:
            summaries = [
                run
                for run in summaries
                if self._matches_args(
                    run.args,
                    self._prune_conditions(
                        (
                            defaults_for_algorithm(run.algorithm)
                            if apply_defaults
                            else {}
                        )
                        | query.args
                    ),
                )
            ]
        return summaries

    def get_run(self, run_id: str) -> RunSummary:
        runs = self.query_runs(RunQuery(run_ids=(run_id,), apply_defaults=False))
        if not runs:
            raise KeyError(f"结果数据库中没有 run_id：{run_id}")
        return runs[0]

    def read_metrics(self, run_id: str) -> list[dict[str, Any]]:
        metrics = self.read_metrics_many((run_id,))
        if run_id not in metrics:
            raise KeyError(f"结果数据库中没有 run_id：{run_id}")
        return metrics[run_id]

    def read_metrics_many(
        self, run_ids: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """一次读取多个运行的数据库指标；不存在的 run_id 不会出现在结果中。"""
        selected_ids = self._values(run_ids)
        if not selected_ids:
            return {}
        if not self.path.is_file():
            raise FileNotFoundError(f"结果数据库不存在：{self.path}")

        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
            if "metrics_json" not in columns:
                rows = connection.execute(
                    "SELECT run_id FROM runs WHERE run_id IN ("
                    + ", ".join("?" for _ in selected_ids)
                    + ")",
                    selected_ids,
                ).fetchall()
                return {row["run_id"]: [] for row in rows}
            rows = connection.execute(
                "SELECT run_id, metrics_json FROM runs WHERE run_id IN ("
                + ", ".join("?" for _ in selected_ids)
                + ")",
                selected_ids,
            ).fetchall()
        return {row["run_id"]: json.loads(row["metrics_json"] or "[]") for row in rows}

    @staticmethod
    def _values(values: Iterable[str] | None) -> tuple[str, ...] | None:
        if values is None:
            return None
        if isinstance(values, str):
            return (values,)
        return tuple(values)

    @staticmethod
    def _add_filter(
        conditions: list[str],
        parameters: list[str],
        column: str,
        values: tuple[str, ...] | None,
    ) -> None:
        if values is None:
            return
        if not values:
            conditions.append("0")
            return
        conditions.append(f"{column} IN ({', '.join('?' for _ in values)})")
        parameters.extend(
            value.lower() if column == "lower(algorithm)" else value for value in values
        )

    @staticmethod
    def _summary(row: sqlite3.Row) -> RunSummary:
        values = dict(row)
        return RunSummary(
            run_id=values["run_id"],
            algorithm=values["algorithm"],
            started_at=values["started_at"],
            ended_at=values["ended_at"],
            status=values["status"],
            max_accuracy=values["max_accuracy"],
            latest_accuracy=values["latest_accuracy"],
            run_path=Path(values["run_path"]),
            args_hash=values["args_hash"],
            args=json.loads(values["args_json"] or "{}"),
            latest_round=values["latest_round"],
        )

    @staticmethod
    def _prune_conditions(conditions: Mapping[str, Any]) -> dict[str, Any]:
        """在使用某种划分方式时，显式忽略互斥的另一种划分参数。"""
        pruned = dict(conditions)
        match pruned.get("partition"):
            case "dirichlet":
                pruned.pop("classes_per_client", None)
            case "pathological":
                pruned.pop("dir", None)
            case "iid":
                pruned.pop("dir", None)
                pruned.pop("classes_per_client", None)

        return pruned

    @staticmethod
    def _matches_args(stored: Mapping[str, Any], conditions: Mapping[str, Any]) -> bool:
        for name, expected in conditions.items():
            if name not in stored:
                return False
            actual = stored[name]
            candidates = expected if isinstance(expected, tuple) else (expected,)
            if not any(
                isinstance(actual, bool) == isinstance(candidate, bool)
                and actual == candidate
                for candidate in candidates
            ):
                return False
        return True
