import os
import torch
from tqdm import tqdm
from create_dataset import create_dir
from training_pipeline import save_checkpoint

def train_vae(model, train_loader, val_loader, device, optimizer: str = 'Adam',
              start_epoch=0, num_epochs=100, learning_rate=1e-3,
              checkpoint_name='vae_checkpoint', best_metric_checkpoint_name='best_vae_model'):
    optimizers = {
        'Adam': torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5),
        'SGD': torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    }

    # Initialize optimizer
    optimizer_label = optimizer
    optimizer = optimizers[optimizer]

    train_losses = []
    val_losses = []

    # Track the best validation loss
    best_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)

            # Forward pass
            outputs, mu, logvar = model(inputs)
            loss = model.loss_function(outputs, inputs, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                inputs = inputs.to(device)
                outputs, mu, logvar = model(inputs)
                loss = model.loss_function(outputs, inputs, mu, logvar)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        models_dir_path = create_dir(f'./model_vae_history_{optimizer_label}')

        checkpoint_path = os.path.join(models_dir_path, checkpoint_name)
        best_metric_checkpoint_path = os.path.join(models_dir_path, best_metric_checkpoint_name)

        # Save regular checkpoint
        save_checkpoint(epoch+1, model, optimizer, val_loss, optimizer_label, f'{checkpoint_path}.pt')

        # Save the best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Best VAE model saved at epoch {epoch+1}")
            save_checkpoint(epoch+1, model, optimizer, val_loss, optimizer_label, f'{best_metric_checkpoint_path}.pt')

    return train_losses, val_losses
