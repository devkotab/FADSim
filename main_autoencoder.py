import os
import time
import csv
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from server import FederatedServer
from clients import FederatedClient
from utils.dataset_partition import partition_dataset
from torchvision import datasets, transforms
from detectors.autoencoder_scorer import AutoencoderScorer
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def evaluate_anomaly_detection(trust_scores, true_labels, thresholds):
    results = {}
    for t in thresholds:
        predicted_labels = [1 if score < t else 0 for score in trust_scores]
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        results[t] = (precision, recall)
    return results


def run_federated_simulation():
    base_dir = "output"
    log_dir = os.path.join(base_dir, "logs")
    plot_dir = os.path.join(base_dir, "plots")
    csv_dir = os.path.join(base_dir, "csv")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    num_clients = 10
    rounds = 30
    dataset = "MNIST"
    malicious_clients = [2, 5, 8]
    true_labels_binary = [
        1 if i in malicious_clients else 0 for i in range(num_clients)
    ]
    thresholds = [0.3, 0.5, 0.7, 0.9]

    client_data = partition_dataset(
        dataset_name=dataset, num_clients=num_clients, non_iid=True
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

    server = FederatedServer()
    clients = [
        FederatedClient(i, client_data[i], is_malicious=(i in malicious_clients))
        for i in range(num_clients)
    ]

    scorer = AutoencoderScorer(
        input_dim=clients[0].get_update_vector().shape[0], train_epochs=2
    )

    accuracy_log = []
    trust_log = []
    duration_log = []
    precision_logs = {t: [] for t in thresholds}
    recall_logs = {t: [] for t in thresholds}

    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1} ---")
        round_start = time.time()

        local_updates = []
        update_vectors = []
        global_weights = server.global_model.state_dict()

        for client in clients:
            weights = client.train(global_weights=global_weights)
            local_updates.append(weights)
            client.model.load_state_dict(weights)
            update_vector = client.get_update_vector()
            update_vectors.append(update_vector)

        trust_scores = scorer.score(update_vectors)
        trust_log.append([round_num + 1] + trust_scores)

        evals = evaluate_anomaly_detection(trust_scores, true_labels_binary, thresholds)
        for t in thresholds:
            precision_logs[t].append([round_num + 1, evals[t][0]])
            recall_logs[t].append([round_num + 1, evals[t][1]])

        scorer.update_reference(update_vectors, labels=true_labels_binary)
        server.aggregate(local_updates, trust_scores=trust_scores)

        acc = evaluate_model(server.global_model, test_loader)
        duration = time.time() - round_start

        accuracy_log.append([round_num + 1, acc])
        duration_log.append([round_num + 1, duration])

        print(f"Round {round_num + 1} Accuracy: {acc:.2f}% - Duration: {duration:.2f}s")
        for t in thresholds:
            print(
                f"  Threshold {t}: Precision={evals[t][0]:.2f}, Recall={evals[t][1]:.2f}"
            )

    # Save logs (CSV format in csv_dir)
    acc_df = pd.DataFrame(accuracy_log, columns=["Round", "Accuracy (%)"])
    trust_df = pd.DataFrame(
        trust_log, columns=["Round"] + [f"Client_{i}" for i in range(num_clients)]
    )
    duration_df = pd.DataFrame(duration_log, columns=["Round", "Duration (s)"])
    acc_df.to_csv(os.path.join(csv_dir, "results_log.csv"), index=False)
    trust_df.to_csv(os.path.join(csv_dir, "trust_log.csv"), index=False)
    duration_df.to_csv(os.path.join(csv_dir, "time_log.csv"), index=False)

    for t in thresholds:
        pd.DataFrame(precision_logs[t], columns=["Round", "Precision"]).to_csv(
            os.path.join(csv_dir, f"precision_log_{t}.csv"), index=False
        )
        pd.DataFrame(recall_logs[t], columns=["Round", "Recall"]).to_csv(
            os.path.join(csv_dir, f"recall_log_{t}.csv"), index=False
        )

    # Plots in plot_dir
    plt.figure()
    sns.lineplot(data=acc_df, x="Round", y="Accuracy (%)", marker="o")
    plt.title("Global Model Accuracy Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "accuracy_plot.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for col in trust_df.columns[1:]:
        sns.lineplot(data=trust_df, x="Round", y=col, label=col)
    plt.title("Client Trust Scores Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Trust Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "trust_scores_plot.png"))
    plt.close()

    for t in thresholds:
        prec_df = pd.DataFrame(precision_logs[t], columns=["Round", "Precision"])
        rec_df = pd.DataFrame(recall_logs[t], columns=["Round", "Recall"])
        plt.figure()
        sns.lineplot(
            data=prec_df, x="Round", y="Precision", label="Precision", marker="o"
        )
        sns.lineplot(data=rec_df, x="Round", y="Recall", label="Recall", marker="x")
        plt.title(f"Precision and Recall at Threshold {t}")
        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"precision_recall_plot_t{int(t*10)}.png"))
        plt.close()

    plt.figure()
    sns.lineplot(data=duration_df, x="Round", y="Duration (s)", marker="o")
    plt.title("Wall-Clock Time Per Round")
    plt.xlabel("Round")
    plt.ylabel("Duration (seconds)")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "duration_plot.png"))
    plt.close()

    # Final round ROC & PR curves
    final_scores = trust_log[-1][1:]
    true_labels = [1 if i in malicious_clients else 0 for i in range(num_clients)]
    scores = np.array(final_scores).astype(float)
    inverted_scores = -scores  # Lower score = more anomalous

    fpr, tpr, _ = roc_curve(true_labels, inverted_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve (Final Round)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "roc_curve_final.png"))
    plt.close()

    prec, rec, _ = precision_recall_curve(true_labels, inverted_scores)
    pr_auc = auc(rec, prec)

    plt.figure()
    plt.plot(rec, prec, label=f"PR Curve (AUC = {pr_auc:.2f})")
    plt.title("Precision-Recall Curve (Final Round)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "pr_curve_final.png"))
    plt.close()


if __name__ == "__main__":
    run_federated_simulation()
