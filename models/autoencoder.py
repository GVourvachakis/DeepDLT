import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self, activation_function: str = 'relu', dropout_strength: float = 0.2, filter_dim: int = 3):
        super(CNNAutoencoder, self).__init__()

        self.dropout_strength = dropout_strength
        
        self.activation_functions = {
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'elu': nn.ELU()
        }

        self.activation_function = self.activation_functions[activation_function]
        self.dropout = nn.Dropout2d(self.dropout_strength)  # defined dropout, inducing randomness

        # Encoder
        self.encoder = nn.Sequential(

            # First Conv block
            #nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(1, 32, kernel_size=filter_dim, stride=1, padding=1 if filter_dim == 3 else 2),  # (64x64 -> 64x64)
            nn.BatchNorm2d(32),
            self.activation_function,
            self.dropout,        # Apply Dropout2d
            nn.MaxPool2d(2, 2),  # (64x64 -> 32x32)

            # Second Conv block
            #nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), 
            nn.Conv2d(32, 64, kernel_size=filter_dim, stride=1, padding=1 if filter_dim == 3 else 2),  # (32x32 -> 32x32)
            nn.BatchNorm2d(64),
            self.activation_function,
            self.dropout,  
            nn.MaxPool2d(2, 2),  # (32x32 -> 16x16)

            # Third Conv block
            #nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(64, 128, kernel_size=filter_dim, stride=1, padding=1 if filter_dim == 3 else 2),  # (16x16 -> 16x16)
            nn.BatchNorm2d(128),
            self.activation_function,
            self.dropout,
            nn.MaxPool2d(2, 2)  # (16x16 -> 8x8)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # (8x8 -> 16x16)
            nn.BatchNorm2d(64),
            self.activation_function,
            #self.dropout,  # Apply Dropout2d

            # In case you prefer nn.Conv2d instead of nn.ConvTransposed2d()
            # ["undo" the spatial compression caused by MaxPool2d]
            #nn.Upsample(scale_factor=2, mode='nearest'), 

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (16x16 -> 32x32)
            nn.BatchNorm2d(32),
            self.activation_function,
            #self.dropout, 
            #nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # (32x32 -> 64x64)
            #nn.BatchNorm2d(1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x): # x^ = gof(x;W);W')
        x = self.encoder(x)
        x = self.decoder(x)
        return x