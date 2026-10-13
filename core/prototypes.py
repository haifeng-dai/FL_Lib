import torch


def extract_prototypes(model, loader, num_classes, feature_dim, device):
    """从本地数据提取按类别平均的表征与样本数。"""
    prototypes = torch.zeros(num_classes, feature_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            features = model.extractor(inputs.to(device, non_blocking=True))
            targets = targets.to(device, non_blocking=True)
            prototypes.index_add_(0, targets, features)
            counts.index_add_(0, targets, torch.ones_like(targets, dtype=counts.dtype))
    nonempty = counts > 0
    prototypes[nonempty] /= counts[nonempty].unsqueeze(1)
    return prototypes.cpu(), counts.cpu()


def aggregate_prototypes(payloads: list, old: torch.Tensor, by_count=False):
    """聚合非空类别原型；可选择按类别样本数加权。"""
    sums = torch.zeros_like(old)
    counts = torch.zeros(old.shape[0], dtype=old.dtype)
    for payload in payloads:
        prototypes = payload["prototypes"]
        local_counts = payload["counts"].to(dtype=old.dtype)
        weights = local_counts if by_count else (local_counts > 0).to(old.dtype)
        sums.add_(prototypes * weights.unsqueeze(1))
        counts.add_(weights)
    result = old.clone()
    nonempty = counts > 0
    result[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    return result
