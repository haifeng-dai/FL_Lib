from collections.abc import Mapping, Sequence

import torch


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """复制为脱离计算图的独立 CPU 模型状态。"""
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def clone_weights(
    weights: list[torch.Tensor] | None,
) -> list[torch.Tensor] | None:
    """复制可跨进程传输的 CPU 权重，未初始化时保留 None。"""
    if weights is None:
        return None
    return [weight.detach().cpu().clone() for weight in weights]


def aggregate_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """按权重聚合状态字典；未给权重时对所有输入等权平均。"""
    if not states:
        raise ValueError("待聚合的状态不能为空")

    if weights is None:
        normalized = [1.0 / len(states)] * len(states)
    else:
        if len(weights) != len(states):
            raise ValueError("状态数量必须与权重数量一致")
        total = sum(weights)
        if total <= 0:
            raise ValueError("聚合权重之和必须为正数")
        normalized = [weight / total for weight in weights]

    first_state = states[0]
    first_keys = first_state.keys()
    for state in states[1:]:
        if state.keys() != first_keys:
            raise ValueError("待聚合的状态必须包含相同的键")
        if any(state[name].shape != first_state[name].shape for name in first_keys):
            raise ValueError("相同状态键对应的张量形状必须一致")

    result = clone_state(dict(first_state))
    for name, tensor in result.items():
        if torch.is_floating_point(tensor):
            tensor.zero_()
            for state, weight in zip(states, normalized, strict=True):
                tensor.add_(state[name], alpha=weight)
    return result
