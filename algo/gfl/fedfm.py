import torch

from core import BaseClient, BaseServer
from core.prototypes import aggregate_prototypes, extract_prototypes


def add_arguments(parser):
    """注册 FedFM 私有超参数。"""
    parser.add_argument("--mu", type=float, default=1.0, help="锚点对齐损失权重")
    parser.add_argument(
        "--start-fm-round",
        type=int,
        default=20,
        help="完成指定轮数的 FedAvg 预训练后启用特征匹配",
    )
    parser.add_argument(
        "--anchor-loss",
        choices=("mse", "prototype"),
        default="prototype",
        help="锚点对齐损失：归一化均方误差或归一化原型对比损失",
    )
    parser.add_argument("--cg-tau", type=float, default=0.1, help="CG 损失的温度参数")


def contrastive_guiding_loss(
    features: torch.Tensor,
    anchors: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    normalized_features = torch.nn.functional.normalize(features, dim=1)
    normalized_anchors = torch.nn.functional.normalize(anchors, dim=1)
    logits = normalized_features @ normalized_anchors.T / temperature
    return torch.nn.functional.cross_entropy(logits, targets)


class FedFMClient(BaseClient):
    """FedFM 两阶段客户端：训练或在全局模型上提取锚点。"""

    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.mu: float = args.mu
        self.feature_dim: int = args.feature_dim
        self.anchor_loss: str = args.anchor_loss
        self.cg_tau: float = args.cg_tau

    def train(self, task):
        if task.payload["mode"] == "extract":
            anchors, counts = extract_prototypes(
                self.model,
                self.train_loader(task.client_id),
                self.num_classes,
                self.feature_dim,
                self.device,
            )
            return self.result(task, 0.0, {"prototypes": anchors, "counts": counts})
        anchors = task.payload["global_anchors"].to(self.device)
        use_fm = task.payload["use_fm"]
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
                target_anchors = anchors[targets]
                valid = target_anchors.abs().sum(dim=1) > 0
                loss = self.ce_loss(logits, targets)
                if use_fm and valid.any():
                    if self.anchor_loss == "mse":
                        normalized_features = torch.nn.functional.normalize(
                            features[valid], dim=1
                        )
                        normalized_anchors = torch.nn.functional.normalize(
                            target_anchors[valid], dim=1
                        )
                        alignment_loss = torch.nn.functional.mse_loss(
                            normalized_features, normalized_anchors
                        )
                    else:
                        alignment_loss = contrastive_guiding_loss(
                            features[valid],
                            anchors,
                            targets[valid],
                            self.cg_tau,
                        )
                    loss = loss + self.mu * alignment_loss
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
        return self.result(task, loss_sum / steps)


class Server(BaseServer):
    """FedFM 服务端，以两次客户端任务更新模型和全局锚点。"""

    client_class = FedFMClient

    def __init__(self, args):
        super().__init__(args)
        self.global_anchors = torch.zeros(self.num_classes, args.feature_dim)
        self.phase = "train"
        self.start_fm_round: int = args.start_fm_round

    def run_round(self):
        self.select_clients()
        self.phase = "train"
        training = self.run_clients()
        self.apply_results(training)
        self.phase = "extract"
        # 原型提取基于聚合后的全局模型，不能覆盖第一阶段的本地训练上传。
        extracted = self.run_clients(cache_uploads=False)
        self.update_anchors(extracted)
        loss = sum(result.loss for result in training.values()) / len(self.selected)
        return loss, self.evaluate()

    def training_payloads(self):
        if self.phase == "extract":
            return {client_id: {"mode": "extract"} for client_id in self.selected}
        return {
            client_id: {
                "mode": "train",
                "global_anchors": self.global_anchors.clone(),
                "use_fm": len(self.accuracies) >= self.start_fm_round,
            }
            for client_id in self.selected
        }

    def apply_results(self, results):
        """处理训练阶段结果，聚合全局模型。"""
        self.aggregate_weighted(results)

    def update_anchors(self, results):
        """处理第二阶段结果，按类别样本数更新全局锚点。"""
        self.global_anchors = aggregate_prototypes(
            [results[client_id].payload for client_id in self.selected],
            self.global_anchors,
            by_count=True,
        )

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "global_anchors": self.global_anchors.clone(),
        }
