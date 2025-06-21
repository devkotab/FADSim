import torch
import torch.nn as nn
import torch.optim as optim


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderScorer:
    def __init__(self, input_dim, hidden_dim=64, train_epochs=30, device="cpu"):
        self.device = device
        self.train_epochs = train_epochs
        self.model = SimpleAutoencoder(input_dim, hidden_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.trained = False

    def normalize_vector(self, vec):
        return (vec - vec.mean()) / (vec.std() + 1e-8)

    def update_reference(self, vectors, labels=None):
        benign_vectors = vectors
        if labels:
            benign_vectors = [v for v, lbl in zip(vectors, labels) if lbl == 0]
        if len(benign_vectors) == 0:
            return
        normed_data = [self.normalize_vector(v) for v in benign_vectors]
        data = torch.stack(normed_data).to(self.device)
        self.model.train()
        for epoch in range(self.train_epochs):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data)
            loss.backward()
            self.optimizer.step()
        self.trained = True

    def score(self, vectors):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for vec in vectors:
                normed = self.normalize_vector(vec).unsqueeze(0).to(self.device)
                recon = self.model(normed)
                loss = self.criterion(recon, normed).item()
                trust_score = 1.0 / (1.0 + loss)
                scores.append(trust_score)
        return scores
