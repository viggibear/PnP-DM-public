import torch

class KNN:
    def __init__(self, k):
        self.k = k
        self.X_train = None

    def fit(self, X):
        self.X_train = X

    def __call__(self, X):
        if self.X_train is None:
            raise RuntimeError("You must fit the model before calling it.")
        # Compute distances from X to the training data
        dist = torch.cdist(X, self.X_train)
        # Get the indices of the k nearest neighbors
        knn_indices = torch.topk(dist, self.k, largest=False).indices
        return knn_indices
