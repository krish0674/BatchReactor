import torch 
import torch.nn as nn
import numpy as np

class Koopman(nn.Module):
    def __init__(self):
        super().__init__()
        # Hardcoded settings
        self.n_states = 2
        self.n_controls = 1
        self.latent_dim = 8
        self.enc_hidden_layer_sizes = [16, 16]  # Two hidden layers of size 16
        self.enc_activation = nn.ReLU()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.n_states + self.n_controls, self.enc_hidden_layer_sizes[0]),
            self.enc_activation,
            nn.Linear(self.enc_hidden_layer_sizes[0], self.enc_hidden_layer_sizes[1]),
            self.enc_activation,
            nn.Linear(self.enc_hidden_layer_sizes[1], self.latent_dim)
        )

        # Koopman operator (linear dynamics in latent space)
        self.Az = nn.Linear(self.latent_dim, self.latent_dim, bias=False)  # State evolution
        self.Au = nn.Linear(self.n_controls, self.latent_dim, bias=False)  # Control influence

        self.decoder = nn.Linear(self.latent_dim, 1)

    def forward(self, x, u):
        """
        x: Current state (dim=2)
        u: Current control (dim=1)
        Output: Predicted next control (dim=1)
        """
        # Ensure batch dimensions
        x, u = self.ensure_batch_shape(x, u)
        xu = torch.cat([x, u], dim=1)  # Combine state and control
        z = self.encoder(xu)  # Encode to latent space
        z_next = self.Az(z) + self.Au(u)  # Predict next latent state
        u_next = self.decoder(z_next)  # Decode to next control
        return u_next
    
    def ensure_batch_shape(self, x, u):
        # Convert to torch tensors if not already
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32)

        # Ensure both are at least 1-dimensional
        if x.dim() == 0:  # If scalar (0D tensor)
            x = x.unsqueeze(0)  # Add batch dimension
        if u.dim() == 0:  # If scalar (0D tensor)
            u = u.unsqueeze(0)  # Add batch dimension

        # Ensure both are 2-dimensional (batch_size, feature_dimension)
        if x.dim() == 1:  # If 1D tensor, add a second dimension (e.g., batch_size)
            x = x.unsqueeze(0)  # Add batch dimension
        if u.dim() == 1:  # If 1D tensor, add a second dimension (e.g., batch_size)
            u = u.unsqueeze(0)  # Add batch dimension

        return x, u

import torch
import torch.nn as nn
import torch.optim as optim

class RBFNN(nn.Module):
    def __init__(self, centers_1, sigma):
        super(RBFNN, self).__init__()
        self.centers_1 = nn.Parameter(torch.tensor(centers_1, dtype=torch.float32), requires_grad=False)
        self.sigma = sigma
        self.linear1 = nn.Linear(len(centers_1), 16, bias=True)
        self.linear2 = nn.Linear(16, 32, bias=True)
        self.linear3 = nn.Linear(32, 16, bias=True)
        self.linear4 = nn.Linear(16, 8, bias=True)
        self.linear5 = nn.Linear(8, 1, bias=True)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ReLU()
        self.activation4 = nn.ReLU()

    def _gaussian(self, x, c):
        return torch.exp(-torch.norm(x - c, dim=-1) ** 2 / (2 * self.sigma ** 2))

    def forward(self, X):
        activations1 = torch.stack([self._gaussian(X, c) for c in self.centers_1], dim=-1)
        activations1 = activations1.view(activations1.size(0), -1)
        x = self.activation1(self.linear1(activations1))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.activation4(self.linear4(x))
        return self.linear5(x)

centers_1 = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [1, 0.5], [0.5, 1], [1, 1]]

model = RBFNN(centers_1=centers_1, sigma=0.5)

