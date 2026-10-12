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
