import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import time

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def get_cifar10_loaders(batch_size=128, num_workers=4, noise_rate=0.4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                   transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                  transform=transform_test)
    
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices, val_indices = indices[:split], indices[split:]
    
    if noise_rate > 0:
        train_labels = np.array(train_dataset.targets)
        noisy_indices = np.random.choice(len(train_indices), 
                                       int(noise_rate * len(train_indices)), 
                                       replace=False)
        for idx in noisy_indices:
            orig_label = train_labels[train_indices[idx]]
            new_label = np.random.choice([i for i in range(10) if i != orig_label])
            train_labels[train_indices[idx]] = new_label
        train_dataset.targets = train_labels.tolist()
    
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        Subset(train_dataset, val_indices),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

class NTKComputer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_ntk(self, train_loader, val_loader, num_samples=None):
        """
        Compute the Neural Tangent Kernel matrix
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_samples: Number of samples to use for computation (None for all samples)
        
        Returns:
            ntk_matrix: NTK matrix
            computation_time: Time taken for computation
        """
        start_time = time.time()
        
        # Collect gradients from training and validation sets
        train_grads = []
        val_grads = []
        
        # Compute training set gradients
        print("Computing training set gradients...")
        for i, (inputs, targets) in enumerate(tqdm(train_loader)):
            if num_samples is not None and i * inputs.size(0) >= num_samples:
                break
                
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            
            # Compute gradients of model outputs with respect to parameters
            grads = torch.autograd.grad(outputs, self.model.parameters(), 
                                      grad_outputs=torch.ones_like(outputs))
            train_grads.append([g.detach().cpu() for g in grads])
        
        # Compute validation set gradients
        print("Computing validation set gradients...")
        for i, (inputs, targets) in enumerate(tqdm(val_loader)):
            if num_samples is not None and i * inputs.size(0) >= num_samples:
                break
                
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            
            # Compute gradients of model outputs with respect to parameters
            grads = torch.autograd.grad(outputs, self.model.parameters(),
                                      grad_outputs=torch.ones_like(outputs))
            val_grads.append([g.detach().cpu() for g in grads])
        
        # Compute NTK matrix
        print("Computing NTK matrix...")
        n_train = len(train_grads)
        n_val = len(val_grads)
        ntk_matrix = torch.zeros(n_train, n_val)
        
        for i in tqdm(range(n_train)):
            for j in range(n_val):
                kernel_value = 0
                for g1, g2 in zip(train_grads[i], val_grads[j]):
                    kernel_value += torch.sum(g1 * g2)
                ntk_matrix[i, j] = kernel_value
        
        computation_time = time.time() - start_time
        
        return ntk_matrix, computation_time
    
    def analyze_ntk(self, ntk_matrix):
        """
        Analyze properties of the NTK matrix
        
        Args:
            ntk_matrix: NTK matrix
        
        Returns:
            analysis_dict: Dictionary containing analysis results
        """
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(ntk_matrix)
        
        # Compute condition number
        condition_number = torch.norm(ntk_matrix) * torch.norm(torch.linalg.inv(ntk_matrix))
        
        # Compute matrix rank
        rank = torch.linalg.matrix_rank(ntk_matrix)
        
        return {
            'eigenvalues': eigenvalues,
            'condition_number': condition_number,
            'rank': rank,
            'shape': ntk_matrix.shape
        }

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = ResNet18().to(device)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=128, noise_rate=0.4
    )
    
    # Create NTK computer
    ntk_computer = NTKComputer(model, device)
    
    # Compute NTK matrix (using fewer samples for demonstration)
    num_samples = 100  # Can be adjusted as needed
    ntk_matrix, computation_time = ntk_computer.compute_ntk(
        train_loader, val_loader, num_samples=num_samples
    )
    
    print(f"\nNTK Computation completed in {computation_time:.2f} seconds")
    print(f"NTK Matrix shape: {ntk_matrix.shape}")
    
    # Analyze NTK matrix
    analysis = ntk_computer.analyze_ntk(ntk_matrix)
    print("\nNTK Analysis:")
    print(f"Matrix shape: {analysis['shape']}")
    print(f"Rank: {analysis['rank']}")
    print(f"Condition number: {analysis['condition_number']:.2f}")
    
    # Save results
    torch.save({
        'ntk_matrix': ntk_matrix,
        'analysis': analysis,
        'computation_time': computation_time
    }, 'ntk_results.pth')

if __name__ == '__main__':
    main()
