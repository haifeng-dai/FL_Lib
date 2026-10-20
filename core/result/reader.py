from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .database import DatabaseReader, RunQuery, RunSummary


class ResultsReader:
    """统一读取运行记录和数据库中的逐轮指标。"""

    def __init__(
        self,
        result_root: str | Path,
        *,
        database: str | Path | None = None,
    ):
        self.root = Path(result_root)
        self.database = DatabaseReader(database or self.root / "results.sqlite")

    def query_runs(self, query: RunQuery | None = None) -> list[RunSummary]:
        return self.database.query_runs(query)

    def get_run(self, run_id: str) -> RunSummary:
        return self.database.get_run(run_id)

    def read_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self.database.read_metrics(run_id)

    def read_metrics_many(
        self, run_ids: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        return self.database.read_metrics_many(run_ids)
