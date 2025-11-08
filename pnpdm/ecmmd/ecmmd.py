import torch

from .knn import KNN

class GaussianKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y):
        diff = X.unsqueeze(1) - Y.unsqueeze(0)
        return torch.exp(-self.sigma * torch.norm(diff, dim=2) ** 2)


class ECMMD:
    def __init__(self, kernel_type, sigma, knn: KNN, l1_lambda: float = 0.0):
        self.sigma = sigma
        self.knn = knn
        self.l1_lambda = l1_lambda
        assert self.knn is not None, "KNN object must be provided."

        if kernel_type == 'gaussian':
            self.kernel = GaussianKernel(self.sigma)
        else:
            raise NotImplementedError(f"Kernel type {kernel_type} is not implemented.")

    @property
    def display_name(self):
        name = f'ecmmd-{self.kernel_type}-sigma={self.sigma}'
        if self.l1_lambda > 0:
            name += f'-l1={self.l1_lambda}'
        return name

    def __call__(self, Z, Y, X):
        batch_size = X.shape[0]
        device = X.device
        N_X = torch.zeros((batch_size, batch_size), device=device)
        row_idx = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, self.knn.k)

        self.knn.fit(X)

        knn_indices = self.knn(X)
        N_X[row_idx, knn_indices] = 1.0

        H = self.kernel(Z, Z) + self.kernel(Y, Y) - self.kernel(Z, Y) - self.kernel(Y, Z)

        loss = torch.sum(N_X * H) / (batch_size * self.knn.k)

        if self.l1_lambda > 0:
            loss += self.l1_lambda * torch.nn.functional.l1_loss(Z, Y)

        return loss
