import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.simple_cnn import SimpleCNN


class FederatedClient:
    def __init__(self, client_id, data, is_malicious=False):
        self.client_id = client_id
        self.data = data
        self.is_malicious = is_malicious
        self.model = SimpleCNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, global_weights=None, epochs=1, batch_size=32):
        if global_weights:
            self.model.load_state_dict(global_weights)

        print(
            f"Client {self.client_id} training {'(malicious)' if self.is_malicious else ''}..."
        )
        x_data, y_data = self.data

        if self.is_malicious:
            y_data = torch.where(y_data == 1, torch.tensor(7), y_data)

        dataset = TensorDataset(x_data, y_data)
        if len(dataset) == 0:
            print(f"Warning: Client {self.client_id} has no data!")
            return self.model.state_dict()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for inputs, labels in loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()

    def get_update_vector(self):
        return torch.cat([v.flatten() for v in self.model.state_dict().values()])
