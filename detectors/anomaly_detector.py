import torch.nn as nn
import torch.nn.functional as F


class AnomalyScorer:
    def __init__(self, method="cosine", ref_vectors=None):
        self.method = method
        self.ref_vectors = ref_vectors or []

    def score(self, update_vector):
        if self.method == "cosine":
            if not self.ref_vectors:
                return 1.0

            sims = [
                F.cosine_similarity(update_vector, ref, dim=0).item()
                for ref in self.ref_vectors
            ]
            avg_sim = sum(sims) / len(sims)
            return max(avg_sim, 0.0)
        else:
            raise NotImplementedError("Only 'cosine' method is supported.")

    def update_reference(self, update_vector):
        self.ref_vectors.append(update_vector.detach())
        if len(self.ref_vectors) > 10:
            self.ref_vectors.pop(0)
