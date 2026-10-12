import torch

from core import BaseClient, BaseServer, clone_state
from core.decentralized import adjacency, metropolis_hastings, mix_states


def add_arguments(parser):
    parser.add_argument("--event-r", type=float, default=1.0)
    parser.add_argument("--bandwidth-mean", type=float, default=1.0)
    parser.add_argument("--bandwidth-std", type=float, default=0.0)


class EFHCClient(BaseClient):
    def train(self, task):
        loss = self.train_supervised(task.client_id)
        state = clone_state(self.model.state_dict())
        squared_norm = torch.stack(
            [
                (state[name] - task.payload["hat_state"][name]).float().square().sum()
                for name in state
            ]
        ).sum()
        size = sum(value.numel() for value in state.values())
        change = torch.sqrt(squared_norm / size).item()
        return self.result(task, loss, {"change": change})


class Server(BaseServer):
    client_class = EFHCClient
    pfl = True

    def __init__(self, args):
        super().__init__(args)
        self.event_r: float = args.event_r
        low: float = (1 - args.bandwidth_std) * args.bandwidth_mean
        high: float = (1 + args.bandwidth_std) * args.bandwidth_mean
        if low <= 0:
            raise ValueError("bandwidth-mean 与 bandwidth-std 必须生成正带宽")
        self.bandwidths = (low + (high - low) * torch.rand(self.num_clients)).tolist()
        self.rho = [1 / bandwidth for bandwidth in self.bandwidths]
        self.graph = adjacency(args)
        self.weights = metropolis_hastings(self.graph).to(self.device)
        self.hat_states = {
            client_id: clone_state(state)
            for client_id, state in self.client_states.items()
        }
        self.client_changes = {client_id: 0.0 for client_id in range(self.num_clients)}
        self.triggered_ids: list[int] = []

    def training_payloads(self):
        return {
            client_id: {"hat_state": self.hat_states[client_id]}
            for client_id in self.selected
        }

    def apply_results(self, results):
        self.update_client_states(results)
        for client_id in self.selected:
            self.client_changes[client_id] = results[client_id].payload["change"]
        gamma = 0.1 * (1 + len(self.accuracies)) ** -0.5
        self.triggered_ids = [
            client_id
            for client_id in range(self.num_clients)
            if self.client_changes[client_id]
            >= self.event_r * self.rho[client_id] * gamma
        ]
        if not self.triggered_ids:
            return

        # A broadcast activates its incident edges. Both endpoints of an
        # active edge aggregate, while inactive edges contribute no update.
        triggered = torch.zeros(self.num_clients, dtype=torch.bool, device=self.device)
        triggered[self.triggered_ids] = True
        graph = self.graph.to(self.device, dtype=torch.bool)
        active_edges = graph & (triggered[:, None] | triggered[None, :])
        active_edges.fill_diagonal_(False)

        event_weights = self.weights.clone()
        event_weights.fill_diagonal_(0)
        event_weights *= active_edges
        event_weights.diagonal().copy_(1 - event_weights.sum(dim=1))

        # Keep the transmitted model as the trigger reference. The subsequent
        # consensus update changes the main model, not the broadcast snapshot.
        broadcast_states = {
            client_id: clone_state(self.client_states[client_id])
            for client_id in self.triggered_ids
        }
        mixed = mix_states(
            [self.client_states[client_id] for client_id in range(self.num_clients)],
            event_weights,
            self.device,
        )
        recipients = torch.where(active_edges.any(dim=1))[0].tolist()
        for client_id in recipients:
            self.client_states[client_id] = mixed[client_id]
        for client_id, state in broadcast_states.items():
            self.hat_states[client_id] = state

    def progress_fields(self):
        return {**super().progress_fields(), "triggered": len(self.triggered_ids)}

    def checkpoint_state(self):
        return {
            **super().checkpoint_state(),
            "graph": self.graph.cpu(),
            "hat_states": self.hat_states,
            "client_changes": self.client_changes,
            "bandwidths": self.bandwidths,
        }
