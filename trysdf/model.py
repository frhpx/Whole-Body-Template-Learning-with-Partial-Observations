import torch
import torch.nn as nn

class TemplateNet(nn.Module):
    def __init__(self, c, hidden_dim=256, omega=30.0):
        super().__init__()
        self.omega = omega
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, c)

        # SIREN initialization
        nn.init.uniform_(self.fc1.weight, -1 / 3, 1 / 3)
        for layer in [self.fc2, self.fc3, self.fc4]:
            nn.init.uniform_(layer.weight,
                             -((6 / hidden_dim) ** 0.5) / omega,
                              ((6 / hidden_dim) ** 0.5) / omega)

    def forward(self, x):
        x = torch.sin(self.omega * self.fc1(x))
        x = torch.sin(self.omega * self.fc2(x))
        x = torch.sin(self.omega * self.fc3(x))
        x = torch.sin(self.omega * self.fc4(x))
        return self.fc5(x)


class DeformNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(3 + latent_dim, hidden_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc4 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)