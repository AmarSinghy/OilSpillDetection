import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class DataLoaderPreparation:
    """Class for preparing data and creating data loaders."""

    def __init__(self, img_size=(256, 256), random_state=52):
        """
        Initialize the DataLoaderPreparation class.

        Args:
            img_size (tuple): Image size (height, width).
            random_state (int): Random state for data splitting.
        """
        self.img_size = img_size
        self.random_state = random_state

    def prepare_data(self, input_images, targets, test_size=0.2):
        """
        Prepare the data by splitting it into train, test, and validation sets.

        Args:
            input_images (ndarray): Input images.
            targets (ndarray): Target data.
            test_size (float): Proportion of the data to be used for the test set.

        Returns:
            tuple: Train inputs, test inputs, train targets, test targets,
                   valid inputs, valid targets.
        """
        # Split the data into train, test, and validation sets
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(
            input_images, targets, test_size=test_size, random_state=self.random_state
        )
        valid_inputs, test_inputs, valid_targets, test_targets = train_test_split(
            test_inputs, test_targets, test_size=0.5, random_state=self.random_state
        )

        # Convert the data to PyTorch tensors and transpose the dimensions
        train_inputs = torch.from_numpy(train_inputs.transpose(0, 3, 1, 2)).type(torch.float32)
        test_inputs = torch.from_numpy(test_inputs.transpose(0, 3, 1, 2)).type(torch.float32)
        train_targets = torch.from_numpy(train_targets.transpose(0, 3, 1, 2)).type(torch.float32)
        test_targets = torch.from_numpy(test_targets.transpose(0, 3, 1, 2)).type(torch.float32)
        valid_inputs = torch.from_numpy(valid_inputs.transpose(0, 3, 1, 2)).type(torch.float32)
        valid_targets = torch.from_numpy(valid_targets.transpose(0, 3, 1, 2)).type(torch.float32)

        return (
            train_inputs, test_inputs, train_targets, test_targets, valid_inputs, valid_targets
        )

    def create_data_loaders(self, train_inputs, train_targets, valid_inputs, valid_targets, batch_size=1):
        """
        Create data loaders for the train and validation sets.

        Args:
            train_inputs (torch.Tensor): Train inputs.
            train_targets (torch.Tensor): Train targets.
            valid_inputs (torch.Tensor): Validation inputs.
            batch_size (int): Batch size.

        Returns:
            tuple: Train data loader, validation data loader.
        """
        # Define your custom dataset class
        class CustomDataset(Dataset):
            """Custom dataset class for inputs and targets."""
            
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets
                """
                Initialize the CustomDataset.

                Args:
                    inputs (torch.Tensor): Input data.
                    targets (torch.Tensor): Target data.
                """
            def __len__(self):
                """
                Get the length of the dataset.

                Returns:
                     int: Length of the dataset.
                """
                return len(self.inputs)
              

            def __getitem__(self, idx):
                """
                Get an item from the dataset.

                Args:
                    idx (int): Index of the item.

                Returns:
                    tuple: Input data and corresponding target data.
                """
                input_data = self.inputs[idx]
                target_data = self.targets[idx]
                return input_data, target_data

        # Create instances of your custom dataset
        train_dataset = CustomDataset(train_inputs, train_targets)
        valid_dataset = CustomDataset(valid_inputs, valid_targets)

        # Create data loaders
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        return trainloader, validloader