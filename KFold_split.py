import pandas as pd
import os
from sklearn.model_selection import KFold

# torch dependencies
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from dataset_loader import LaserDataset

def create_cross_validation_loaders(csv_file, n_splits=3, batch_size=32, num_workers=4):
    """
    Creates DataLoader instances for 5-fold cross-validation.
    
    Args:
        csv_file (str): Path to the CSV file with dataset information.
        n_splits (int): Number of folds for cross-validation.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
    
    Returns:
        List[Tuple[DataLoader, DataLoader]]: List of train and validation DataLoader pairs for each fold.
    """
    # Load dataset
    dataset = LaserDataset(csv_file)
    dataset_size = len(dataset)
    
    # Prepare K-Fold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_loaders = []

    for train_idx, val_idx in kf.split(range(dataset_size)):
        # Create subsets for train and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create DataLoader instances for train and validation
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        fold_loaders.append((train_loader, val_loader))

    return fold_loaders

# Example usage
csv_file_path = "./datasets/data_with_labels_csv/training_data.csv"
cross_val_loaders = create_cross_validation_loaders(csv_file_path)

# Accessing the DataLoaders for a specific fold
for fold, (train_loader, val_loader) in enumerate(cross_val_loaders):
    print(f"Fold {fold + 1}:")
    for images, labels in train_loader:
        print(f"Train batch size: {images.size()}")
        break
    for images, labels in val_loader:
        print(f"Validation batch size: {images.size()}")
        break