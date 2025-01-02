import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNVariationalAutoencoder(nn.Module):
    def __init__(self, activation_function: str = 'relu', dropout_strength: float = 0.2, latent_dim: int = 128):
        
        super(CNNVariationalAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.dropout_strength = dropout_strength

        self.activation_functions = {
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'leakyrelu': nn.LeakyReLU(0.1,inplace=True)
        }

        self.activation_function = self.activation_functions[activation_function]
        self.dropout = nn.Dropout2d(self.dropout_strength)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            self.activation_function,
            self.dropout,
            nn.MaxPool2d(2, 2),  # (64x64 -> 32x32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.activation_function,
            self.dropout,
            nn.MaxPool2d(2, 2),  # (32x32 -> 16x16)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            self.activation_function,
            self.dropout,
            nn.MaxPool2d(2, 2)  # (16x16 -> 8x8)
        )

        # Latent space (mean and log-variance layers)
        self.fc_mu = nn.Linear(128 * 8 * 8, self.latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, self.latent_dim)

        # Decoder (from latent space to feature map size before decoding)
        self.fc_decoder = nn.Linear(self.latent_dim, 128 * 8 * 8)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # (8x8 -> 16x16)
            nn.BatchNorm2d(64),
            self.activation_function,

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (16x16 -> 32x32)
            nn.BatchNorm2d(32),
            self.activation_function,

            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # (32x32 -> 64x64)
            nn.Sigmoid()  # Output between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample epsilon from standard normal
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # Flatten for fully connected layers

        # Latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Decoder
        x = self.fc_decoder(z)
        x = x.view(-1, 128, 8, 8)  # Reshape to feature map
        x = self.decoder(x)
        return x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        VAE Loss = Reconstruction Loss + KL Divergence
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_divergence