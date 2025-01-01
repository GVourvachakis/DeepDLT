import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from create_dataset import create_dir

from typing import Type
# Type -> Class data type
        
def save_checkpoint(epoch, model, optimizer: torch.optim.Adam | torch.optim.SGD,
                    loss, optimizer_label: str, file_path: str):
    
    # type(optimizer) = torch.optim.sgd.SGD | torch.optim.adam.Adam
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, file_path)
    print(f"Checkpoint saved to {file_path}, epoch: {epoch} with optimizer {optimizer_label}")

def load_checkpoint(model, optimizer: str, file_path: str, lr: float = 1e-3):

    optimizers = { 
            'Adam': optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5),
            'SGD': optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            }
    
    # Initialize optimizer (select from user)
    optimizer = optimizers[optimizer]

    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: epoch={epoch}, loss={loss:.4f}")
    return model, optimizer, epoch, loss

def normalize_image(img):
        # Normalize to [0,1] range
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min)

# Training Function
def train_model(model, train_loader, val_loader, device, optimizer: str = 'Adam', 
                criterion=nn.MSELoss(), start_epoch=0, num_epochs=100,learning_rate=1e-3,
                checkpoint_name='model_checkpoint', best_metric_checkpoint_name='best_model'):

    optimizer_label = optimizer

    optimizers = { 
            'Adam': optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5),
            'SGD': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            }
    
    # Initialize optimizer (select from user)
    optimizer = optimizers[optimizer]
    # optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_losses = []
    val_losses = []
    psnr_values = []
    ssim_values = []

    # Variables to track the best metrics
    best_loss = float('inf')
    best_psnr = 0
    best_ssim = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # --Validation phase--
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0

        # Validation loop 
        with torch.no_grad():
            for inputs, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)

                # Compute PSNR and SSIM
                inputs_np = inputs.cpu().numpy()
                outputs_np = outputs.cpu().numpy()

                for inp, out in zip(inputs_np, outputs_np):
                    inp = inp.squeeze()  # Remove the channel dimension (grayscale)
                    out = out.squeeze()

                    # Normalize images for PSNR and SSIM calculation 
                    # [for compatibility with the literature]
                    inp_norm = normalize_image(inp)
                    out_norm = normalize_image(out)
                    
                    # Compute PSNR
                    val_psnr += psnr(inp, out, data_range=inp.max() - inp.min())
                    
                    # Compute SSIM with normalized images
                    val_ssim += ssim(inp_norm, out_norm,
                                    data_range=1.0,  # Using 1.0 as data range since images are normalized
                                    win_size=5) # appropriate window size for 64x64 images

        val_loss /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)

        val_losses.append(val_loss)
        psnr_values.append(val_psnr)
        ssim_values.append(val_ssim)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        models_dir_path = create_dir(f'./models_history_{optimizer_label}')

        checkpoint_path = os.path.join(models_dir_path, checkpoint_name)

        best_metric_checkpoint_path = os.path.join(models_dir_path, best_metric_checkpoint_name)

        # Save regular checkpoint
        save_checkpoint(epoch+1, model, optimizer, val_loss, optimizer_label, f'{checkpoint_path}.pt')

        # Save the best model based on loss
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Best model based on Loss saved at epoch {epoch+1}")
            save_checkpoint(epoch+1, model, optimizer, val_loss, optimizer_label, f'{best_metric_checkpoint_path}_loss.pt')
            

        # Save the best model based on PSNR
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(epoch+1, model, optimizer, val_loss, optimizer_label, f'{best_metric_checkpoint_path}_psnr.pt')
            print(f"Best model based on PSNR saved at epoch {epoch+1}")

        # Save the best model based on SSIM
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            save_checkpoint(epoch+1, model, optimizer, val_loss, optimizer_label, f'{best_metric_checkpoint_path}_ssim.pt')
            print(f"Best model based on SSIM saved at epoch {epoch+1}")

        # Save checkpoints based on different metrics
        #save_model(val_loss, val_psnr, val_ssim, epoch, model, optimizer, optimizer_label)
    
    return train_losses, val_losses, psnr_values, ssim_values