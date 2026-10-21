import hashlib
import json
import logging
import math
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

MetricValue = int | float | None
MetricsRow = dict[str, Any]

BASE_METRIC_COLUMNS = ("round", "train_loss", "accuracy", "elapsed_seconds")
DB_SNAPSHOT_INTERVAL = 50
DB_RETRY_DELAYS = (0.2, 0.5, 1.0)


class Results:
    """管理单次运行的指标归档和总结果库记录。"""

    def __init__(self):
        self.root = Path("results")
        self.database = self.root / "results.sqlite"
        self.run_id: str | None = None
        self._args: dict[str, Any] | None = None
        self._started: datetime | None = None
        self._run_directory: Path | None = None
        self._metrics_file: Path | None = None
        self._log_file: Path | None = None
        self._checkpoint_directory: Path | None = None
        self._metric_columns: tuple[str, ...] = ()
        self._required_metric_columns: tuple[str, ...] = ()
        self._metric_history: list[dict[str, Any]] = []
        self._args_json = ""
        self._args_hash = ""
        self._schema_ready = False
        self._logger = logging.getLogger(__name__)

    def start(
        self,
        args,
        *,
        algorithm_script: str | Path,
        client_metric: Literal["global_acc", "client_macro_acc"] = "client_macro_acc",
        extra_metric_columns: tuple[str, ...] = (),
    ) -> str:
        """创建本次运行的归档，并在总库登记运行参数和 running 状态。"""
        if self.run_id is not None:
            raise RuntimeError("Results 实例已经启动一次运行")

        self._args = vars(args).copy()
        self._args_json = json.dumps(self._args, sort_keys=True, ensure_ascii=False)
        encoded = json.dumps(self._args, sort_keys=True, ensure_ascii=False).encode()
        args_hash = hashlib.blake2s(encoded, digest_size=3).hexdigest()
        self._args_hash = args_hash
        started = datetime.now().astimezone()
        run_id = (
            f"{started:%H%M%S}-{args.algo.lower()}-trial-{args.trial:03d}-{args_hash}"
        )
        run_directory = (
            self.root / f"{started:%Y}" / f"{started:%m}" / f"{started:%d}" / run_id
        )
        algorithm_script = Path(algorithm_script)
        if not algorithm_script.is_file():
            raise FileNotFoundError(f"算法脚本不存在：{algorithm_script}")

        required_columns = BASE_METRIC_COLUMNS
        if getattr(args, "eval_clients", False):
            required_columns += (client_metric,)
        columns = required_columns + extra_metric_columns
        if len(columns) != len(set(columns)):
            raise ValueError("指标名称重复")

        run_directory.mkdir(parents=True, exist_ok=False)
        try:
            shutil.copy2(algorithm_script, run_directory / algorithm_script.name)
        except Exception:
            shutil.rmtree(run_directory)
            raise

        checkpoint_directory = run_directory / "checkpoints"
        checkpoint_directory.mkdir(exist_ok=False)
        metrics_file = run_directory / "metrics.jsonl"
        log_file = run_directory / "run.log"
        (run_directory / "args.json").write_text(
            json.dumps(self._args, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metrics_file.touch(exist_ok=False)

        self.run_id = run_id
        self._started = started
        self._run_directory = run_directory
        self._metrics_file = metrics_file
        self._log_file = log_file
        self._checkpoint_directory = checkpoint_directory
        self._metric_columns = columns
        self._required_metric_columns = required_columns
        self._metric_history = []

        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._try_database_write(
            "登记运行",
            lambda connection: self._insert_run(
                connection,
                status="running",
                ended_at="",
                metrics=[],
                max_accuracy=0.0,
                latest_accuracy=0.0,
            ),
        )
        return run_id

    def append(self, metrics: MetricsRow) -> None:
        """追加数值指标和可选的结构化数据。"""
        self._require_started()
        assert self._metrics_file is not None

        unknown = set(metrics) - set(self._metric_columns)
        missing = set(self._required_metric_columns) - set(metrics)
        if missing:
            raise ValueError("缺少必需的指标：" + ", ".join(sorted(missing)))

        row: dict[str, Any] = {}
        for name in self._metric_columns:
            value = metrics.get(name)
            if value is None:
                if name in self._required_metric_columns:
                    raise ValueError(f"必需指标 {name!r} 不能留空")
                row[name] = ""
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"指标 {name!r} 必须是数值或 None")
            if not math.isfinite(value):
                raise ValueError(f"指标 {name!r} 必须是有限数值")
            row[name] = value

        structured_values = {}
        for name in unknown:
            value = metrics[name]
            if not isinstance(value, (list, dict)):
                raise TypeError(
                    f"未声明的指标 {name!r} 必须是列表或字典结构；"
                    "数值指标需通过 extra_metric_columns 声明"
                )
            _validate_json_value(value, name)
            structured_values[name] = json.loads(
                json.dumps(value, ensure_ascii=False, allow_nan=False)
            )

        history_row = {
            name: row[name] if row[name] != "" else None
            for name in self._metric_columns
        }
        history_row.update(structured_values)
        encoded_row = json.dumps(
            history_row,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        with self._metrics_file.open("a", encoding="utf-8") as file:
            file.write(encoded_row + "\n")
        self._metric_history.append(history_row)
        if int(row["round"]) % DB_SNAPSHOT_INTERVAL == 0:
            self._persist_snapshot()

    def finish(
        self,
        status: Literal["completed", "failed", "test"],
    ) -> None:
        """将内存中的完整指标快照和最终运行状态写入总库。"""
        self._require_started()
        assert self.run_id is not None

        metrics = self._metric_history
        accuracies = [
            float(row["accuracy"])
            for row in metrics
            if isinstance(row.get("accuracy"), (int, float))
        ]
        ended = datetime.now().astimezone()
        try:
            self._try_database_write(
                "保存最终结果",
                lambda connection: self._insert_run(
                    connection,
                    status=status,
                    ended_at=ended.isoformat(),
                    metrics=metrics,
                    max_accuracy=max(accuracies, default=0.0),
                    latest_accuracy=accuracies[-1] if accuracies else 0.0,
                ),
            )
        finally:
            self._close_logging()

    def configure_logging(self, level: str) -> logging.Logger:
        """配置只写入当前运行目录的日志记录器。"""
        self._require_started()
        assert self._log_file is not None
        assert self.run_id is not None

        logger = logging.getLogger(f"fl_lib.{self.run_id}")
        logger.setLevel(level)
        logger.handlers.clear()
        handler = logging.FileHandler(self._log_file, encoding="utf-8")

        class LocalTimeFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                return (
                    datetime.fromtimestamp(record.created)
                    .astimezone()
                    .isoformat(timespec="milliseconds")
                )

        handler.setFormatter(
            LocalTimeFormatter("%(asctime)s | %(levelname)s\n%(message)s")
        )
        logger.addHandler(handler)
        self._logger = logger
        return logger

    def _close_logging(self) -> None:
        """刷新并关闭当前运行的文件日志，支持进程内连续运行多个实验。"""
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)
            handler.flush()
            handler.close()

    def save_checkpoint(self, round_index: int, checkpoint: dict[str, Any]) -> None:
        """保存框架生成的检查点，不向训练代码暴露目录结构。"""
        self._require_started()
        assert self._checkpoint_directory is not None

        import torch

        torch.save(
            checkpoint,
            self._checkpoint_directory / f"round_{round_index}.pt",
        )

    @property
    def startup_log_file(self) -> Path:
        self._require_started()
        assert self._log_file is not None
        return self._log_file

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=0.25)
        connection.execute("PRAGMA busy_timeout = 250")
        return connection

    def _persist_snapshot(self) -> None:
        if not self._metric_history:
            return
        latest = self._metric_history[-1]
        accuracies = [
            float(row["accuracy"])
            for row in self._metric_history
            if isinstance(row.get("accuracy"), (int, float))
        ]
        self._try_database_write(
            f"保存第 {latest['round']} 轮快照",
            lambda connection: self._insert_run(
                connection,
                status="running",
                ended_at="",
                metrics=self._metric_history,
                max_accuracy=max(accuracies, default=0.0),
                latest_accuracy=float(latest["accuracy"]),
            ),
        )

    def _insert_run(
        self,
        connection: sqlite3.Connection,
        *,
        status: str,
        ended_at: str,
        metrics: list[dict[str, Any]],
        max_accuracy: float,
        latest_accuracy: float,
    ) -> None:
        assert self.run_id is not None
        assert self._args is not None
        assert self._started is not None
        assert self._run_directory is not None
        connection.execute(
            """INSERT INTO runs (
                run_id, algorithm, started_at, ended_at, status, max_accuracy,
                latest_accuracy, run_path, args_hash, args_json, metrics_json,
                latest_round
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                ended_at = excluded.ended_at,
                status = excluded.status,
                max_accuracy = excluded.max_accuracy,
                latest_accuracy = excluded.latest_accuracy,
                metrics_json = excluded.metrics_json,
                latest_round = excluded.latest_round""",
            (
                self.run_id,
                self._args.get("algo", ""),
                self._started.isoformat(),
                ended_at,
                status,
                max_accuracy,
                latest_accuracy,
                str(self._run_directory),
                self._args_hash,
                self._args_json,
                json.dumps(metrics, sort_keys=True, ensure_ascii=False),
                int(metrics[-1]["round"]) if metrics else 0,
            ),
        )

    def _try_database_write(self, label: str, write) -> bool:
        """执行短事务；数据库忙时有限重试，最终放弃而不影响归档写入。"""
        for attempt in range(len(DB_RETRY_DELAYS) + 1):
            connection = None
            try:
                connection = self._connect()
                with connection:
                    if not self._schema_ready:
                        self._ensure_schema(connection)
                    write(connection)
                self._schema_ready = True
                return True
            except sqlite3.Error as exc:
                if attempt == len(DB_RETRY_DELAYS):
                    self._logger.warning(
                        "SQLite %s失败，已跳过本次数据库写入：%s", label, exc
                    )
                    return False
                time.sleep(DB_RETRY_DELAYS[attempt])
            finally:
                if connection is not None:
                    connection.close()
        return False

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                algorithm TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                status TEXT NOT NULL,
                max_accuracy REAL NOT NULL,
                latest_accuracy REAL NOT NULL,
                run_path TEXT NOT NULL,
                args_hash TEXT NOT NULL,
                args_json TEXT NOT NULL DEFAULT '{}',
                metrics_json TEXT NOT NULL DEFAULT '[]',
                latest_round INTEGER NOT NULL DEFAULT 0
            )"""
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        if "args_json" not in columns:
            connection.execute(
                "ALTER TABLE runs ADD COLUMN args_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "metrics_json" not in columns:
            connection.execute(
                "ALTER TABLE runs ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "latest_round" not in columns:
            connection.execute(
                "ALTER TABLE runs ADD COLUMN latest_round INTEGER NOT NULL DEFAULT 0"
            )

    def _require_started(self) -> None:
        if self.run_id is None:
            raise RuntimeError("调用此方法前必须先调用 Results.start()")


def _validate_json_value(value: Any, path: str) -> None:
    """确保结构化指标能无损写入 JSONL 和 SQLite JSON 字段。"""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} 中不能包含非有限数值")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} 的对象键必须是字符串")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} 包含不支持的 JSON 类型：{type(value).__name__}")
