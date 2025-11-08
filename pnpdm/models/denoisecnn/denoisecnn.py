import torch
from torch import nn

class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        # Encoder: 64x64 -> 64x8x8
        # 64 -> 32 -> 16 -> 8 (3 MaxPool layers, each dividing by 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )

        # fc1: processes the encoded features (64 * 8 * 8)
        self.fc1 = nn.Sequential(nn.Linear(64 * 8 * 8, 64 * 8 * 8),
                                 nn.ReLU())

        # fc2: processes concatenated features and eta (65 * 8 * 8)
        self.fc2 = nn.Sequential(nn.Linear(65 * 8 * 8, 65 * 8 * 8),
                                 nn.ReLU())

        # Decoder: 65x8x8 -> 64x64
        # 8 -> 16 -> 32 -> 64 (3 ConvTranspose layers, each multiplying by 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(65, 64, kernel_size=2, stride=2),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),   # 32 -> 64
            nn.Sigmoid()
        )

    def forward(self, x, eta):
        # Encode: (batch, 1, 64, 64) -> (batch, 64, 8, 8)
        x = self.encoder(x)

        # Flatten: (batch, 64, 8, 8) -> (batch, 64*8*8)
        x = x.view(-1, 64 * 8 * 8)

        # Process through fc1
        x = self.fc1(x)
        x = self.fc1(x)

        # Reshape back: (batch, 64*8*8) -> (batch, 64, 8, 8)
        x = x.view(-1, 64, 8, 8)

        # Concatenate with eta
        # eta should be (batch, 1, 8, 8) or (batch, 8, 8)
        if len(eta.shape) == 3:
            # eta is (batch, 8, 8) -> add channel dim
            eta = eta.unsqueeze(1)
            x = torch.cat((x, eta), dim=1)  # (batch, 65, 8, 8)
        elif len(eta.shape) == 2:
            # eta is (8, 8) -> add batch and channel dims
            eta = eta.unsqueeze(0).unsqueeze(0)
            x = torch.cat((x, eta.expand(x.size(0), -1, -1, -1)), dim=1)

        # Flatten: (batch, 65, 8, 8) -> (batch, 65*8*8)
        x = x.view(-1, 65 * 8 * 8)

        # Process through fc2
        x = self.fc2(x)
        x = self.fc2(x)

        # Reshape: (batch, 65*8*8) -> (batch, 65, 8, 8)
        x = x.view(-1, 65, 8, 8)

        # Decode: (batch, 65, 8, 8) -> (batch, 1, 64, 64)
        x = self.decoder(x)

        return x
