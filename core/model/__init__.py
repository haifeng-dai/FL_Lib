import torch

from .cnn import CNN


class Model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.extractor: torch.nn.Sequential = model.extractor
        self.classifier: torch.nn.Linear = model.classifier
        self.feature_dim: int = self.classifier.in_features

    def forward(self, inputs: torch.Tensor, req_feat=False):
        feature: torch.Tensor = self.extractor(inputs)
        output: torch.Tensor = self.classifier(feature)
        if req_feat:
            return feature, output
        else:
            return output


def build_model(
    name: str,
    dataset: str,
    num_classes: int,
) -> Model:
    """构建模型。"""
    if name == "cnn":
        base_model = CNN(1 if dataset == "mnist" else 3, num_classes)
    else:
        raise ValueError(f"不支持的模型：{name}")
    return Model(base_model)
