import torch
from models.simple_cnn import SimpleCNN


class FederatedServer:
    def __init__(self):
        self.global_model = SimpleCNN()

    def aggregate(self, client_weights, trust_scores=None):
        print("Aggregating model updates...")

        if trust_scores is None:
            trust_scores = [1.0 for _ in client_weights]

        total = sum(trust_scores)
        trust_scores = [score / total for score in trust_scores]

        new_state = {}
        for key in self.global_model.state_dict().keys():
            new_state[key] = sum(
                trust_scores[i] * client_weights[i][key]
                for i in range(len(client_weights))
            )
        self.global_model.load_state_dict(new_state)
