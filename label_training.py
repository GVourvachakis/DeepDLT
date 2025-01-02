import torch.nn as nn
import os; import torch
from dataset_loader import LaserDataset
from torch.utils.data import DataLoader
from training_pipeline import train_model
from inference import visualize_reconstruction

def train_and_evaluate_kfold(model_class, fold_dir, num_folds, device, features, 
                             optimizer = "Adam", num_workers=4, criterion=nn.MSELoss(), num_epochs=10, learning_rate=1e-3):
    # fold_train_losses = []
    # fold_val_losses = []
    # fold_test_losses = []

    feature_results = {}
    print(f"Performing {num_folds}-Fold Cross-Validation for features: {features}...\n")

    for feature in features:
        print(f"Starting cross-validation for feature: {feature}")
        fold_train_losses = []
        fold_val_losses = []
        fold_test_losses = []

        for fold in range(1, num_folds + 1):
            print(f"Processing Fold {fold}/{num_folds} for feature: {feature}...")
            
            # Construct paths for the current feature and fold
            train_path = os.path.join(fold_dir, feature, f'fold_{fold}', f'{feature}_train.csv')
            val_path = os.path.join(fold_dir, feature, f'fold_{fold}', f'{feature}_val.csv')
            test_path = os.path.join(fold_dir, feature, f'fold_{fold}', f'{feature}_test.csv')

            # Load datasets
            train_dataset = LaserDataset(train_path)
            val_dataset = LaserDataset(val_path)
            test_dataset = LaserDataset(test_path)

            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

            # Initialize model
            model = model_class(activation_function='relu', dropout_strength=0.2).to(device)

            # Train model
            train_losses, val_losses, _, _ = train_model(
                model, train_loader, val_loader, device, 
                optimizer=optimizer, num_epochs=num_epochs, learning_rate=learning_rate,
                checkpoint_name='model_checkpoint_labels', best_metric_checkpoint_name='best_model_labels')
            
            # Evaluate model on test set
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    test_loss += criterion(inputs, outputs).item() * inputs.size(0)

            test_loss /= len(test_loader.dataset)
            fold_test_losses.append(test_loss)

            # Log fold results
            print(f"Fold {fold}/{num_folds} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Test Loss: {test_loss:.4f}")
            fold_train_losses.append(train_losses)
            fold_val_losses.append(val_losses)

            # Visualize reconstruction for the fold
            visualize_reconstruction(test_loader, model, device, num_images=5)

        # Aggregate results for the feature
        avg_test_loss = sum(fold_test_losses) / num_folds
        feature_results[feature] = {
            "train_losses": fold_train_losses,
            "val_losses": fold_val_losses,
            "test_losses": fold_test_losses,
            "avg_test_loss": avg_test_loss
        }

        print(f"Feature '{feature}' - Average Test Loss across folds: {avg_test_loss:.4f}\n")

    # Compare features by average test loss
    print("Comparison of reconstruction difficulty across features:")
    for feature, results in feature_results.items():
        print(f"  {feature}: Average Test Loss = {results['avg_test_loss']:.4f}")

    return feature_results