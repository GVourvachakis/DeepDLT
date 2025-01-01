import os
import pandas as pd
# from sklearn.model_selection import train_test_split
from create_dataset import create_dir

from sklearn.model_selection import StratifiedKFold

def k_fold_cross_validation(data, label, output_dir, fold=5):
    f"""
    Subroutine for {fold}-fold cross-validation splitting with stratification.

    Args:
        data (pd.DataFrame): The input dataset.
        label (str): The label column to base stratification on.
        output_dir (str): Directory to save cross-validation splits.
        fold (int): fold hyper-parameter (default is 5).
    """
    # Prepare output directory
    create_dir(output_dir)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

    for i, (train_index, val_test_index) in enumerate(skf.split(data, data[label])):
        print(f"Processing fold {i + 1}...")
        # Split indices into training and validation+test sets
        train_data = data.iloc[train_index].reset_index(drop=True)
        val_test_data = data.iloc[val_test_index].reset_index(drop=True)

        # Further split val_test_data into validation and test sets (50/50 split)
        val_size = len(val_test_data) // 2
        val_data = val_test_data[:val_size].reset_index(drop=True)
        test_data = val_test_data[val_size:].reset_index(drop=True)

        # Save the datasets
        fold_dir = create_dir(os.path.join(output_dir, f'fold_{i + 1}'))
        train_path = os.path.join(fold_dir, f'{label}_train.csv')
        val_path = os.path.join(fold_dir, f'{label}_val.csv')
        test_path = os.path.join(fold_dir, f'{label}_test.csv')

        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)

        print(f"Fold {i + 1}:")
        print(f"  Training set size: {len(train_data)}")
        print(f"  Validation set size: {len(val_data)}")
        print(f"  Test set size: {len(test_data)}\n")


# def k_fold_cross_validation(data, label, output_dir, fold=5):
#     f"""
#     Subroutine for {fold}-fold cross-validation splitting.

#     Args:
#         data (pd.DataFrame): The input dataset.
#         label (str): The label column to base stratification on.
#         output_dir (str): Directory to save cross-validation splits.
#         fold (int): fold hyper-parameter (default is 5).
#     """
#     # Prepare output directory
#     create_dir(output_dir)

#     # Split the original data into 5 folds
#     data_folds = []
#     fold_size = len(data) // fold
#     for i in range(fold):
#         start_idx = i * fold_size
#         end_idx = (i + 1) * fold_size if i < fold-1 else len(data)  # Ensure no data is left out
#         data_folds.append(data[start_idx:end_idx])

#     for i in range(fold):
#         print(f"Processing fold {i + 1}...")
#         # Combine all folds except the current one for training
#         train_folds = [data_folds[j] for j in range(fold) if j != i]
#         train_data = pd.concat(train_folds).reset_index(drop=True)

#         # Use the current fold for validation and test splitting
#         val_test_data = data_folds[i].reset_index(drop=True)
#         val_size = len(val_test_data) // 2

#         val_data = val_test_data[:val_size].reset_index(drop=True)
#         test_data = val_test_data[val_size:].reset_index(drop=True)

#         # Save the datasets
#         fold_dir = create_dir(os.path.join(output_dir, f'fold_{i + 1}'))
#         train_path = os.path.join(fold_dir, f'{label}_train.csv')
#         val_path = os.path.join(fold_dir, f'{label}_val.csv')
#         test_path = os.path.join(fold_dir, f'{label}_test.csv')

#         train_data.to_csv(train_path, index=False)
#         val_data.to_csv(val_path, index=False)
#         test_data.to_csv(test_path, index=False)

#         print(f"Fold {i + 1}:")
#         print(f"  Training set size: {len(train_data)}")
#         print(f"  Validation set size: {len(val_data)}")
#         print(f"  Test set size: {len(test_data)}\n")

def main_cross_validation(csv_output_path, labels, output_dir, fold=5):
    f"""
    Execute {fold}-fold cross-validation for multiple labels.

    Args:
        csv_output_path (str): Path to the directory containing the combined dataset.
        labels (list): List of labels for stratification.
        output_dir (str): Output directory for cross-validation datasets.
        fold (int): fold hyper-parameter (default is 5).
    """

    # Load and combine datasets
    combined_df = pd.concat([
        pd.read_csv(os.path.join(csv_output_path, 'training_data.csv')),
        pd.read_csv(os.path.join(csv_output_path, 'validation_data.csv')),
        pd.read_csv(os.path.join(csv_output_path, 'testing_data.csv'))
    ]).reset_index(drop=True)

    # Perform cross-validation for each label
    for label in labels:
        if label not in combined_df.columns:
            print(f"Warning: Label '{label}' not found in the dataset. Skipping...")
            continue

        # Check if the label is continuous and bin if necessary
        if pd.api.types.is_numeric_dtype(combined_df[label]):
            print(f"Label '{label}' is continuous. Binning values for stratification...")
            combined_df[f'{label}'] = pd.qcut(combined_df[label], q=5, labels=False, duplicates='drop')
            #strat_label = f'{label}'  #f'{label}_binned'
       # else:
        
        strat_label = label

        print(f"Performing {fold}-fold cross-validation for label: {label}")
        print("-" * 50)
        label_dir = create_dir(os.path.join(output_dir, label))
        k_fold_cross_validation(combined_df, strat_label, label_dir, fold=fold)


# Example usage
if __name__ == "__main__":
    csv_output_path = "./datasets/data_with_labels_csv"
    output_dir = "./datasets/label_aware_splitting_data"
    labels_to_split = ['angle', 'PP1', 'NP', 'EP1']

    main_cross_validation(csv_output_path, labels_to_split, output_dir, fold=3)