from pathlib import Path

_COMMON = {
    "dataset": "cifar10",
    "model": "cnn",
    "num_rounds": 1000,
    "num_epochs": 10,
    "lr": 0.01,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "batch_size": 64,
    "num_clients": 20,
    "join_ratio": 1.0,
    "seed": 42,
    "test_ratio": 0.2,
    "feature_dim": 512,
    "pre_train": None,
}

DEFAULT_QUERY_ARGS = {
    "gfl": {**_COMMON, "partition": "dirichlet", "dir": 0.1, "eval_clients": False},
    "pfl": {**_COMMON, "partition": "dirichlet", "dir": 0.1},
    "dfl": {
        **_COMMON,
        "partition": "dirichlet",
        "dir": 0.1,
        "adj_type": "ring",
        "edge_p": 0.2,
        "k_small_world": 4,
        "m_scale_free": 2,
    },
}


def defaults_for_algorithm(algorithm: str) -> dict:
    """按项目中的算法目录取得该家族的公共默认查询参数。"""
    algorithm_root = Path(__file__).resolve().parents[2] / "algo"
    for family, defaults in DEFAULT_QUERY_ARGS.items():
        if (algorithm_root / family / f"{algorithm.lower()}.py").is_file():
            return defaults
    return {}
