import matplotlib.pyplot as plt
import torch
from typing import List
       
def plotting(train_losses: List[float], val_losses: List[float], 
             psnr_values: List[float] = [], ssim_values: List[float] = [], model_label: str = "Autoencoder") -> None:
    """
    Plot training and validation losses, PSNR values, and SSIM values.
    """

    # Plot training and validation losses
    plt.figure(figsize=(18, 6))

    # Plot training and validation losses
    plt.subplot(1, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    
    if model_label == "Autoencoder":
        # Plot PSNR
        plt.subplot(1, 3, 2)
        plt.plot(epochs, psnr_values, label='PSNR', color='green')
        plt.xlabel("Epochs")
        plt.ylabel("PSNR (dB)")
        plt.title("PSNR Values")
        plt.legend()

        # Plot SSIM
        plt.subplot(1, 3, 3)
        plt.plot(epochs, ssim_values, label='SSIM', color='purple')
        plt.xlabel("Epochs")
        plt.ylabel("SSIM")
        plt.title("SSIM Values")
        plt.legend()

    plt.tight_layout()
    plt.show()

# Visualization Function
def visualize_reconstruction(data_loader, model, model_label = 'Autoencoder', device='cpu', num_images=10):
    """
    Visualizes original and reconstructed images from an Autoencoder.

    Args:
        data_loader (DataLoader): DataLoader for the dataset.
        model_label (str): Model label for selection. Default 'Autoencoder'
        model (nn.Module): Trained (V)AE model.
        device (torch.device): Device to perform computations on.
        num_images (int): Number of images to visualize. Default is 10.
    """

    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            
            if model_label == 'Autoencoder':
                outputs = model(inputs)
            
            elif model_label == "Variational Autoencoder":
                outputs,_,_  = model(inputs) # Assuming VAE returns (output, mu, logvar)
            
            else: print("model_label = {'Autoencoder', 'Variational Autoencoder'}")

            # Ensure we visualize up to the available number of images
            num_images = min(num_images, inputs.size(0))

            # Visualize Original and Reconstructed Images
            # Create subplots
            fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 6), constrained_layout=True)

            # Plot original images
            for i in range(num_images):
                ax = axes[0, i]
                ax.imshow(inputs[i].cpu().squeeze(), cmap='gray', aspect='equal')
                ax.axis('off')
                if i == 0:
                    ax.set_ylabel("Original", fontsize=14)

            # Plot reconstructed images
            for i in range(num_images):
                ax = axes[1, i]
                ax.imshow(outputs[i].cpu().squeeze(), cmap='gray', aspect='equal')
                ax.axis('off')
                if i == 0:
                    ax.set_ylabel("Reconstructed", fontsize=14)

            # Show the plot
            plt.suptitle("Reconstruction Visualization", fontsize=16)
            plt.show()
            break
 

 # ========================  VAE INFERENCE  ============================

def inference_vae(model, data_loader, device):
    
    model.eval()
    reconstructed_images = []
    original_images = []
    latent_vectors = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs, mu, logvar = model(inputs)

            # Collect outputs and latent variables
            reconstructed_images.append(outputs.cpu())
            original_images.append(inputs.cpu())
            latent_vectors.append(mu.cpu())  # Use `mu` as the latent representation

    return reconstructed_images, original_images, latent_vectors

def sample_from_latent_space(model, num_samples, latent_dim, device):
    model.eval()
    sampled_images = []

    with torch.no_grad():
        # Sample latent vectors from a normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)

        # Decode latent vectors to generate images
        generated_images = model.decode(z)
        sampled_images.extend(generated_images.cpu())

    return sampled_images

#Inference:
# reconstructed_images, original_images, latent_vectors = inference_vae(model, test_loader, device)

#Sampling:
# sampled_images = sample_from_latent_space(model, num_samples=10, latent_dim=model.latent_dim, device=device)


