# native python dependencies
import pandas as pd
from PIL import Image
import numpy as np
import os

# torch dependencies
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

# custom dependencies
from create_dataset import create_dataset, create_data_with_labels_csv, create_dir


class LaserDataset(Dataset):
    def __init__(self, csv_file, transform=None, denoise=False, add_noise=False):
        """
        Args:
            csv_file (str): Path to CSV file with image paths and labels
            transform (callable, optional): Optional transform to be applied on a sample 
                                            (e.g., Fourier transform)
            denoise (bool, optional): Optional denoising mechanism
            add_noise (bool, optional): Optinal Gaussian noise injection
        """
        self.data = pd.read_csv(csv_file)
        self.denoise = denoise
        self.add_noise = add_noise
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]) #or transform

        # Ensure the columns have correct types
        self.data['angle'] = self.data['angle'].astype(float)
        self.data['PP1'] = self.data['PP1'].astype(str)
        self.data['NP'] = self.data['NP'].astype(int)
        self.data['EP1'] = self.data['EP1'].astype(int)

        # Map "PP1" to integer values for categorical encoding
        self.pp1_mapping = {'V': 0, 'H': 1, 'D': 2}
        self.data['PP1_encoded'] = self.data['PP1'].map(self.pp1_mapping)

    def __len__(self):
        return len(self.data)
    
    def apply_fft_denoise(self, image, threshold=0.1):
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        mask = np.ones_like(f_shift)
        mask[f_shift < threshold * np.max(np.abs(f_shift))] = 0
        return np.real(np.fft.ifft2(np.fft.ifftshift(f_shift * mask)))

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = self.data.iloc[idx]['image_path']

        # Load image using PIL
        with Image.open(img_path) as img:
            image = img.convert('L') # Convert to grayscale by 
                                    #ITU-R 601-2 luma transform: L = R * 299/1000 + G * 587/1000 + B * 114/1000)

        # Apply denoising if enabled
        if self.denoise:
            image = self.apply_fft_denoise(image)

        # Add Gaussian noise if enabled
        if self.add_noise:
            noise = np.random.normal(0, 0.1, image.shape)
            image = np.clip(image + noise, 0, 255)

        # Ensure image is in PIL format before applying transforms
        if self.transform:
            image = self.transform(image)

        # Get labels
        labels = {
                    'angle': torch.tensor(self.data.iloc[idx]['angle'], dtype=torch.float32),
                    'PP1': torch.tensor(self.data.iloc[idx]['PP1_encoded'], dtype=torch.long),
                    'NP': torch.tensor(self.data.iloc[idx]['NP'], dtype=torch.int32),
                    'EP1': torch.tensor(self.data.iloc[idx]['EP1'], dtype=torch.int32)
                 }

        # Convert labels to tensors
        # labels = {k: torch.tensor(v, dtype=torch.float32) for k, v in labels.items()}

        return image, labels

def prepare_and_load_data(input_dirs, base_path, output_dir_images, excel_path, csv_output_path,
                           cropped_dim=64, num_workers=4, batch_size=32):
    """
    Complete pipeline for data preparation and loading by 
    creating DataLoader instances for training, validation and testing
    Sets train,test,val dataloaders 
    """

    # Generate augmented dataset
    create_dataset(input_dirs, output_dir_images, base_path, desired_dim=cropped_dim)

    # Create CSV files with labels
    excel_data = pd.read_excel(excel_path, engine = 'openpyxl')
    create_data_with_labels_csv(excel_data, output_dir_images, csv_output_path)

    # Create dataloaders
    print("Creating DataLoaders...\n")
    print("Training DataLoader...")
    train_loader = DataLoader(
        LaserDataset(os.path.join(csv_output_path, 'training_data.csv')),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    print("Validation DataLoader...")
    val_loader = DataLoader(
        LaserDataset(os.path.join(csv_output_path, 'validation_data.csv')),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print("Testing DataLoader...")
    test_loader = DataLoader(
        LaserDataset(os.path.join(csv_output_path, 'testing_data.csv')),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #print(type(test_loader))
    
    return train_loader, val_loader, test_loader

def test_laser_dataset(csv_path, batch_size=32, denoise=False, add_noise=False):
    """Quick test function for LaserDataset"""

    # Verify CSV path
    try:
        print("CSV head:")
        print(pd.read_csv(csv_path).head())
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None, None

    # Initialize dataset and dataloader
    dataset = LaserDataset(csv_path, denoise=denoise, add_noise=add_noise)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # row-wise sampling

    # Test first batch
    images, labels = next(iter(dataloader))
    print(images.shape)
    print(labels)
    # Print results
    print(f"Batch shape: {images.shape}")
    print("\nLabel sample:")
    for k, v in labels.items():
        if v.dim() == 0:  # Single value
            print("dim of data is zero")
            print(f"{k}: {v.item()}")
        else:  # Tensor
            print(f"{k}: {v[0].item()}")
    
    return images, labels


def main(check_loader: bool = False) -> None:
    
    # Define paths
    input_dirs = [
                    '2020-4-30 tuning ripple period',
                    '2020-6-9 Crossed polarized',
                    'Paper Data/Double pulses',
                    'Paper Data/Repetition 6p & 2p 29-4-2020',
                    'Paper Data/Single pulses 2p',
                    'Paper Data/Single pulses 4 and half 6',
                    'Paper Data/Repetition 6p & 2p 29-4-2020/Details'
                 ]
    
    base_path = "./images"
    excel_path = "./images/all_images.xlsx"
    csv_output_path = "./datasets/data_with_labels_csv"

    dim = 64 # set dimensions of augmented images

    images_path = f'./datasets/2023_im_dataset_{dim}x{dim}'
    output_dir_images = create_dir(images_path)

    # Run complete pipeline
    # train_loader, val_loader, test_loader = prepare_and_load_data(
    #     input_dirs,
    #     base_path,
    #     output_dir_images,
    #     excel_path,
    #     csv_output_path,
    #     cropped_dim=dim
    # )

    prepare_and_load_data(
        input_dirs,
        base_path,
        output_dir_images,
        excel_path,
        csv_output_path,
        cropped_dim=dim
    )

    # test on testing_data.csv:
    if check_loader:
        csv_path = "./datasets/data_with_labels_csv/testing_data.csv"
        images, labels = test_laser_dataset(csv_path, denoise=False, add_noise=False)


if __name__ == "__main__":
    main(check_loader=False)