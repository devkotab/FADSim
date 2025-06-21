import torch
from torchvision import datasets, transforms
import numpy as np
import random


def partition_dataset(dataset_name="MNIST", num_clients=10, non_iid=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if dataset_name == "MNIST":
        dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data = dataset.data
    targets = dataset.targets
    client_data = []

    if non_iid:
        num_shards = num_clients * 2
        shard_size = len(data) // num_shards
        idxs = np.arange(len(data))
        labels = targets.numpy()
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        shards = [
            torch.tensor(idxs[i * shard_size : (i + 1) * shard_size])
            for i in range(num_shards)
        ]
        shard_indices = list(range(num_shards))
        random.shuffle(shard_indices)

        for i in range(num_clients):
            client_indices = []
            while len(client_indices) < 2:
                if len(shard_indices) == 0:
                    raise RuntimeError(
                        "Not enough shards to assign non-empty data to all clients."
                    )
                shard_id = shard_indices.pop()
                if len(shards[shard_id]) > 0:
                    client_indices.extend(shards[shard_id].tolist())

            imgs = data[client_indices].unsqueeze(1).float() / 255.0
            lbls = targets[client_indices]
            client_data.append((imgs, lbls))
    else:
        indices = torch.randperm(len(data))
        split_size = len(data) // num_clients
        for i in range(num_clients):
            idx = indices[i * split_size : (i + 1) * split_size]
            imgs = data[idx].unsqueeze(1).float() / 255.0
            lbls = targets[idx]
            client_data.append((imgs, lbls))

    return client_data
