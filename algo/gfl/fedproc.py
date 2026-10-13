import torch

from core import BaseClient, BaseServer
from core.prototypes import aggregate_prototypes, extract_prototypes


def prototype_loss(
    features: torch.Tensor, prototypes: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """计算以类别原型为分类器的余弦交叉熵损失。"""
    normalized_features = torch.nn.functional.normalize(features, dim=1)
    normalized_prototypes = torch.nn.functional.normalize(prototypes, dim=1)
    logits = normalized_features @ normalized_prototypes.T
    return torch.nn.functional.cross_entropy(logits, targets)


class FedProcClient(BaseClient):
    """带全局类别原型对比约束的客户端。"""

    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.feature_dim: int = args.feature_dim

    def train(self, task):
        prototypes = task.payload["global_prototypes"].to(self.device)
        alpha = task.payload["alpha"]
        optimizer = self.build_optimizer()
        self.model.train()
        loss_sum, steps = 0.0, 0
        for _ in range(self.num_epochs):
            for inputs, targets in self.train_loader(task.client_id):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                features = self.model.extractor(inputs)
                logits = self.model.classifier(features)
                loss = (1.0 - alpha) * self.ce_loss(
                    logits, targets
                ) + alpha * prototype_loss(features, prototypes, targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
        local_prototypes, counts = extract_prototypes(
            self.model,
            self.train_loader(task.client_id),
            self.num_classes,
            self.feature_dim,
            self.device,
        )
        return self.result(
            task,
            loss_sum / steps,
            {"prototypes": local_prototypes, "counts": counts},
        )


class Server(BaseServer):
    """FedProc 服务端，维护全局类别原型。"""

    client_class = FedProcClient

    def __init__(self, args) -> None:
        super().__init__(args)
        self.feature_dim: int = args.feature_dim
        self.global_prototypes = torch.zeros(self.num_classes, self.feature_dim)
        self.alpha = 1.0

    def training_payloads(self):
        completed_rounds = len(self.accuracies)
        self.alpha = (
            0.0 if completed_rounds == 0 else 1.0 - completed_rounds / self.num_rounds
        )
        return {
            client_id: {
                "global_prototypes": self.global_prototypes.clone(),
                "alpha": self.alpha,
            }
            for client_id in self.selected
        }

    def apply_results(self, results):
        """按 FedAvg 权重聚合模型，并更新全局类别原型。"""
        self.aggregate_weighted(results)
        self.global_prototypes = aggregate_prototypes(
            [results[client_id].payload for client_id in self.selected],
            self.global_prototypes,
            by_count=True,
        )

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "global_prototypes": self.global_prototypes.clone(),
        }
