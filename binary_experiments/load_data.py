from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def prepare_mnist_binary_classification(digit1, digit2, device='cuda'):
    """
    Prepares MNIST dataset for binary classification between two digits.

    Args:
        digit1 (int): First digit (label 0)
        digit2 (int): Second digit (label 1)
        device (str or torch.device): Device to move tensors to

    Returns:
        train_images, train_labels, val_images, val_labels, test_images, test_labels
    """
    # Define preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Filter and relabel train data
    images, labels = zip(*[(img, label) for img, label in dataset if label in [digit1, digit2]])
    images = torch.stack([img.view(-1) for img in images])
    labels = torch.tensor([1 if label == digit2 else 0 for label in labels])
    labels = F.one_hot(labels, num_classes=2).float()

    # Filter and relabel test data
    test_images, test_labels = zip(*[(img, label) for img, label in test_dataset if label in [digit1, digit2]])
    test_images = torch.stack([img.view(-1) for img in test_images])
    test_labels = torch.tensor([1 if label == digit2 else 0 for label in test_labels])
    test_labels = F.one_hot(test_labels, num_classes=2).float()

    # Train-validation split
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # Move to device
    return (
        train_images.to(device), train_labels.to(device),
        val_images.to(device), val_labels.to(device),
        test_images.to(device), test_labels.to(device)
    )


def create_data_loaders(
        train_X, train_y,
        val_X, val_y,
        test_X, test_y,
        train_batch_size=None,
        val_batch_size=None,
        test_batch_size=None,
        shuffle=False
):
    """
    Wrap tensors into DataLoaders with optional batch sizes.

    Args:
        train_X, train_y, val_X, val_y, test_X, test_y: torch tensors
        train_batch_size (int or None): Batch size for training set
        val_batch_size (int or None): Batch size for validation set
        test_batch_size (int or None): Batch size for test set
        shuffle (bool): Whether to shuffle the training data

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    test_dataset = TensorDataset(test_X, test_y)

    if train_batch_size is None:
        train_batch_size = len(train_dataset)
    if val_batch_size is None:
        val_batch_size = len(val_dataset)
    if test_batch_size is None:
        test_batch_size = len(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

