import matplotlib.pyplot as plt
import torch
from typing import List

       
def plotting(train_losses: List[float], val_losses: List[float], 
             psnr_values: List[float], ssim_values: List[float]) -> None:
    
    # Plot training and validation losses
    plt.figure(figsize=(12, 6))

    # Plot losses
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()

    # Plot PSNR and SSIM
    plt.subplot(1, 2, 2)
    plt.plot(epochs, psnr_values, label='PSNR')
    plt.plot(epochs, ssim_values, label='SSIM')
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("PSNR and SSIM")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Visualization Function
def visualize_reconstruction(data_loader, model, device, num_images=10):
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Visualize Original and Reconstructed Images
            # Set the figure size to accommodate square images
            plt.figure(figsize=(20, 12))
            #plt.gray()

            # Display original images
            for i in range(min(num_images, inputs.size(0))):
                plt.subplot(2, num_images, i + 1)
                plt.imshow(inputs[i].cpu().squeeze(), cmap='gray', aspect='equal')  # Keep images square
                plt.axis('off')

            # Display reconstructed images
            for i in range(min(num_images, outputs.size(0))):
                plt.subplot(2, num_images, num_images + i + 1)
                plt.imshow(outputs[i].cpu().squeeze(), cmap='gray', aspect='equal')  # Keep images square
                plt.axis('off')

            # Adjust layout to increase space between rows
            #plt.subplots_adjust(hspace=0.05)  # Increase vertical space between rows
            plt.tight_layout()  # Adjust layout to prevent overlap
            plt.show()
            break
 