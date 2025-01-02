# from utils import create_dir
import pandas as pd 
import random
import cv2
import os 
import re
import numpy as np

# Function to create a directory if it doesn't exist
def create_dir(directory):
    '''Helper function to create a directory if it doesn't exist.'''
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def create_dataset(in_dirs, out_dir_images, path, desired_dim=64, height=960, width=1280, number_of_batches=100):
    '''
    Generates a dataset by cropping, augmenting, and normalizing portions from images in specified directories.

    Args:
        in_dirs (list): List of input directories containing images.
        out_dir_images (str): Path to the output directory where processed images will be saved.
        height (int): Height of the original images.
        width (int): Width of the original images.
        desired_dim (int): Desired dimension for each cropped image (assumes square crop).
        path (str): Base path for the input directories.
        number_of_batches (int): Number of random crops to generate per image.

    Returns:
        None: The function saves cropped and augmented images to the specified output directory.
    '''
    print("Generating dataset with cropped images...\n")

    instance=1 # new directory's identifier based on the current iteration
    for _dir in in_dirs:
        file = os.listdir(os.path.join(path, _dir))

        # Replace '/' with '_' in directory names for compatibility
        replaced_dir = _dir.replace('/', '_') if 'Paper Data' in _dir else _dir
        
        for im in file:
            if im.endswith('.png'):

                # Create a new directory for each image
                tmp_dir_image = create_dir(os.path.join(out_dir_images, str(instance)))
                os.makedirs(tmp_dir_image, exist_ok=True)

                # Read the image in grayscale
                #image = cv2.imread(os.path.join(path+'/'+_dir,im),cv2.IMREAD_GRAYSCALE)
                image = cv2.imread(os.path.join(path, _dir, im), cv2.IMREAD_GRAYSCALE)
                
                # Ensure the image has the expected height and width
                if image.shape[0] != height or image.shape[1] != width:
                    print(f"\nimage {image} is resized to {width}x{height}\n")
                    image = cv2.resize(image, (width, height))  # Resize if the dimensions are not correct
                

                # Crop, augment, and save parts of the image
                for n in range(number_of_batches):
                    
                    # Generate random coordinates for cropping
                    x_cord = random.randint(0, height - desired_dim)
                    y_cord = random.randint(0, width - desired_dim)
                    
                    # Crop the image
                    cropped_image = image[x_cord:x_cord + desired_dim, y_cord:y_cord + desired_dim]
                    
                    # Intensity augmentation: adjust brightness randomly
                    brightness_factor = random.uniform(0.8, 1.2)  # Adjust brightness by ±20%
                    augmented_image = np.clip(cropped_image * brightness_factor, 0, 255).astype(np.uint8)
                    
                    # Normalize the image to [0, 1] range
                    #augmented_image = augmented_image / 255.0
                    
                    # Save the augmented and normalized image
                    batch_name = f"{replaced_dir}_{im[:-4]}_{n+1}.bmp"
                    save_path = os.path.join(tmp_dir_image, batch_name)
                    
                    # Save augmented image in save_path
                    cv2.imwrite(save_path, augmented_image)
            instance+=1

def create_data_with_labels_csv(data_xlsx, images_path, path_to_csv_data, 
                                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    '''
    Creates labeled datasets in CSV format for training, validation, and testing from provided image data.

    Args:
        data_xlsx (DataFrame): DataFrame containing image names and associated labels.
        images_path (str): Directory path where the images are stored.
        path_to_csv_data (str): Directory path to save the generated CSV files.

    Returns:
        None: The function generates CSV files containing image paths and labels for each dataset type.
    '''
    data = os.listdir(images_path)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    split_idx_1 = int(len(data) * train_ratio)
    split_idx_2 = int(len(data) * (train_ratio + val_ratio))
    
    training_data = data[:split_idx_1]
    validation_data = data[split_idx_1:split_idx_2]
    testing_data = data[split_idx_2:]
    
    print(f"Creating csv files with (train_ratio, val_ratio, test_ratio) = ({train_ratio}, {val_ratio}, {test_ratio})\n")

    # Helper function to process and label data
    def process_data(data_subset, dataset_type):
        data_with_labels = []
        for dir_num in data_subset:
            new_path = os.path.join(images_path, dir_num)
            for image in os.listdir(new_path):
                image_name = re.sub(r'_\d+.bmp', '', image)
                image_name = image_name.replace("_", "/") + ".bmp"

                # Extract labels from the xlsx file
                angle = data_xlsx.loc[data_xlsx['Names'] == image_name, 'angle [deg]'].iloc[0]
                PP1 = data_xlsx.loc[data_xlsx['Names'] == image_name, 'PP1'].iloc[0]
                PP1 = data_xlsx.loc[data_xlsx['Names'] == image_name, 'PP2'].iloc[0] if PP1 in ['0', 0] else PP1
                NP = data_xlsx.loc[data_xlsx['Names'] == image_name, 'NP'].iloc[0]
                EP1 = data_xlsx.loc[data_xlsx['Names'] == image_name, 'EP1 [μJ]'].iloc[0]
                
                image_path = os.path.join(new_path, image)
                data_with_labels.append({'image_path': image_path, 'angle': angle, 'PP1': PP1, 'NP': NP, 'EP1': EP1})

        # Save to CSV
        df = pd.DataFrame(data_with_labels)
        df.to_csv(os.path.join(path_to_csv_data, f'{dataset_type}_data.csv'), index=False)
        
    # Create directories for CSV data
    create_dir(path_to_csv_data)

    # Process and save each data subset
    process_data(training_data, 'training')
    process_data(validation_data, 'validation')
    process_data(testing_data, 'testing')


if __name__ == '__main__':
    
    path = os.getcwd()
    
    # Dimensions of the original and desired cropped images    
    height = 960
    width = 1280
    num_batches = 100
    desired_dim = 64

    
    # Directories containing the original image data
    in_dirs =[
                '2020-4-30 tuning ripple period',
                '2020-6-9 Crossed polarized',
                'Paper Data/Double pulses',
                'Paper Data/Repetition 6p & 2p 29-4-2020',
                'Paper Data/Single pulses 2p',
                'Paper Data/Single pulses 4 and half 6',
                'Paper Data/Repetition 6p & 2p 29-4-2020/Details'
             ]
    
    # Create the dataset and labeled data
    images_path = f'./datasets/2023_im_dataset_{desired_dim}x{desired_dim}'
    out_dir_images = create_dir(images_path)

    # in_dirs, out_dir_images, path, height=960, width=1280, desired_dim=64, number_of_batches=100
    create_dataset(in_dirs, out_dir_images, path = './images',\
                   desired_dim=desired_dim, height=height, width=width, number_of_batches=num_batches)
    
    data_xlsx = pd.read_excel("./images/all_images.xlsx", engine = 'openpyxl')
    create_data_with_labels_csv(data_xlsx, images_path, path_to_csv_data = './datasets/data_with_labels_csv')