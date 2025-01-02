# models/__init__.py
# This can be empty or can expose specific modules/classes
from .autoencoder import CNNAutoencoder  # This will expose the Autoencoder
from .vae import CNNVariationalAutoencoder # This will expose the Variational Autoencoder