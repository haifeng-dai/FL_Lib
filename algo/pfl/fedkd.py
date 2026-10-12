import torch

from core import BaseClient, BaseServer, clone_state


def add_arguments(parser):
    parser.add_argument("--lr-g", type=float, default=0.01)
    parser.add_argument("--energy-start", type=float, default=0.95)
    parser.add_argument("--energy-end", type=float, default=0.98)


def compress(tensor, energy):
    if tensor.ndim not in (2, 4):
        return tensor.detach().cpu()
    shape = tensor.shape
    matrix = tensor.reshape(shape[0], -1) if tensor.ndim == 4 else tensor
    try:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        try:
            u, s, vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
        except RuntimeError:
            return tensor.detach().cpu()
    total_energy = s.square().sum()
    if total_energy == 0:
        return tensor.detach().cpu()
    cumulative_energy = torch.cumsum(s.square(), dim=0)
    reached = cumulative_energy > energy * total_energy
    rank = (
        len(s)
        if not reached.any()
        else int(torch.searchsorted(reached.int(), 1).item() + 1)
    )
    return {
        "u": u[:, :rank].detach().cpu(),
        "s": s[:rank].detach().cpu(),
        "vh": vh[:rank].detach().cpu(),
        "original_shape": shape,
        "is_compressed": True,
    }


def decompress(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if not value.get("is_compressed"):
        raise ValueError("未知的 FedKD 压缩参数格式")
    matrix = value["u"].to(device) @ (
        torch.diag(value["s"].to(device)) @ value["vh"].to(device)
    )
    return matrix.view(value["original_shape"])


class FedKDClient(BaseClient):
    def __init__(self, args, device, num_classes):
        super().__init__(args, device, num_classes)
        self.lr_g: float = args.lr_g
        self.mentees = {}

    def train(self, task):
        client_id = task.client_id
        if client_id not in self.mentees:
            initial_state = task.payload["initial_mentee"]
            if initial_state is None:
                raise RuntimeError("FedKD 客户端缺少初始 mentee 状态")
            self.mentees[client_id] = clone_state(initial_state)
        for update in task.payload["server_updates"]:
            for name, value in update.items():
                self.mentees[client_id][name].add_(decompress(value, "cpu"))
        global_state = {
            k: value.to(self.device) for k, value in self.mentees[client_id].items()
        }
        global_model = self.copy_model(global_state).requires_grad_(True).train()
        align = torch.nn.Linear(self.feature_dim, self.feature_dim, bias=False).to(
            self.device
        )
        if task.payload["align"] is not None:
            align.load_state_dict(task.payload["align"])
        local_opt = self.build_optimizer()
        global_opt = torch.optim.SGD(
            global_model.parameters(),
            lr=self.lr_g,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        align_opt = torch.optim.SGD(
            align.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        total = 0.0
        steps = 0
        for _ in range(self.num_epochs):
            for x, y in self.train_loader(task.client_id):
                x, y = x.to(self.device), y.to(self.device)
                f = self.model.extractor(x)
                fg = global_model.extractor(x)
                o = self.model.classifier(f)
                og = global_model.classifier(fg)
                scale = self.ce_loss(o, y).item() + self.ce_loss(og, y).item() + 1e-8
                loss = (
                    self.ce_loss(o, y)
                    + torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(o, 1),
                        torch.nn.functional.softmax(og.detach(), 1),
                        reduction="batchmean",
                    )
                    / scale
                    + torch.nn.functional.mse_loss(f, align(fg.detach())) / scale
                )
                lossg = (
                    self.ce_loss(og, y)
                    + torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(og, 1),
                        torch.nn.functional.softmax(o.detach(), 1),
                        reduction="batchmean",
                    )
                    / scale
                    + torch.nn.functional.mse_loss(f.detach(), align(fg)) / scale
                )
                local_opt.zero_grad()
                global_opt.zero_grad()
                align_opt.zero_grad()
                loss.backward(retain_graph=True)
                lossg.backward()
                local_opt.step()
                global_opt.step()
                align_opt.step()
                total += loss.item()
                steps += 1
        return self.result(
            task,
            total / steps,
            {
                "update": {
                    k: compress(
                        value.detach() - global_state[k], task.payload["energy"]
                    )
                    if torch.is_floating_point(value)
                    else torch.zeros_like(value, device="cpu")
                    for k, value in global_model.state_dict().items()
                },
                "align": clone_state(align.state_dict()),
                "sync_count": task.payload["sync_count"],
            },
        )


class Server(BaseServer):
    client_class = FedKDClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.energy_start: float = args.energy_start
        self.energy_end: float = args.energy_end
        if not 0.0 < self.energy_start <= self.energy_end <= 1.0:
            raise ValueError("FedKD 能量阈值须满足 0 < energy-start <= energy-end <= 1")
        self.initial_mentee = clone_state(self.model.state_dict())
        self.downlink_updates = []
        self.client_sync_versions = {i: 0 for i in range(self.num_clients)}
        self.client_initialized = {i: False for i in range(self.num_clients)}
        self.align = {i: None for i in range(self.num_clients)}

    def current_energy(self):
        progress = min((len(self.accuracies) + 1) / self.num_rounds, 1.0)
        return self.energy_start + progress * (self.energy_end - self.energy_start)

    def training_payloads(self):
        energy = self.current_energy()
        return {
            i: {
                "initial_mentee": (
                    self.initial_mentee if not self.client_initialized[i] else None
                ),
                "server_updates": self.downlink_updates[self.client_sync_versions[i] :],
                "sync_count": len(self.downlink_updates),
                "align": self.align[i],
                "energy": energy,
            }
            for i in self.selected
        }

    def run_clients(self, *, cache_uploads=True):
        self.client_task_batch += 1
        results = self.pool.run_affined(self.build_tasks())
        if cache_uploads:
            self.update_uploaded_states(results)
        return results

    def apply_results(self, results):
        self.update_client_states(results)
        for i in self.selected:
            self.align[i] = results[i].payload["align"]
            self.client_initialized[i] = True
            self.client_sync_versions[i] = results[i].payload["sync_count"]
        total = sum(results[i].num_samples for i in self.selected)
        current_state = self.model.state_dict()
        energy = self.current_energy()
        global_update = {}
        state = {}
        for name in self.model.state_dict():
            average_update = (
                torch.stack(
                    [
                        decompress(results[i].payload["update"][name], self.device)
                        * results[i].num_samples
                        for i in self.selected
                    ]
                ).sum(dim=0)
                / total
            ).cpu()
            if torch.is_floating_point(current_state[name]):
                packed_update = compress(average_update, energy)
                global_update[name] = packed_update
                state[name] = current_state[name] + decompress(packed_update, "cpu")
            else:
                global_update[name] = torch.zeros_like(current_state[name]).cpu()
                state[name] = current_state[name]
        self.model.load_state_dict(state)
        self.downlink_updates.append(global_update)

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "initial_mentee": self.initial_mentee,
            "downlink_updates": self.downlink_updates,
            "client_sync_versions": self.client_sync_versions,
            "client_initialized": self.client_initialized,
            "align": self.align,
        }
