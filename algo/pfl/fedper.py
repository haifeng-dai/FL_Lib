from core import BaseClient, BaseServer, aggregate_states


class FedPerClient(BaseClient):
    """同时训练共享特征提取器和私有分类头。"""

    def train(self, task):
        loss = self.train_supervised(task.client_id)
        return self.result(task, loss)


class Server(BaseServer):
    """仅聚合特征提取器，保留每个客户端的分类头。"""

    client_class = FedPerClient
    pfl = True

    def apply_results(self, results):
        self.update_client_states(results)
        total = sum(results[client_id].num_samples for client_id in self.selected)
        weights = [
            results[client_id].num_samples / total for client_id in self.selected
        ]
        bodies = [
            {
                name.removeprefix("extractor."): value
                for name, value in results[client_id].state.items()
                if name.startswith("extractor.")
            }
            for client_id in self.selected
        ]
        body = aggregate_states(bodies, weights)
        self.model.extractor.load_state_dict(body)
        for state in self.client_states.values():
            for name, value in body.items():
                state[f"extractor.{name}"] = value.clone()
