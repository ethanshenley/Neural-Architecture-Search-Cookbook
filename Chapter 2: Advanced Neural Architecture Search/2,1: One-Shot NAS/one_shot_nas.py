import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Dict, Tuple
import random

class MixedLayer(nn.Module):
    """Layer containing all possible operations."""
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.ops = nn.ModuleDict({
            'conv3x3': nn.Sequential(
                nn.Conv2d(C_in, C_out, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(C_out),
                nn.ReLU()
            ),
            'conv5x5': nn.Sequential(
                nn.Conv2d(C_in, C_out, 5, stride=stride, padding=2, bias=False),
                nn.BatchNorm2d(C_out),
                nn.ReLU()
            ),
            'maxpool3x3': nn.Sequential(
                nn.MaxPool2d(3, stride=stride, padding=1),
                nn.BatchNorm2d(C_in),
                nn.ReLU()
            ),
            'avgpool3x3': nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1),
                nn.BatchNorm2d(C_in),
                nn.ReLU()
            ),
            'skip': nn.Identity() if stride == 1 and C_in == C_out else None
        })
        
        self.op_names = list(self.ops.keys())

    def forward(self, x: torch.Tensor, op_name: str = None) -> torch.Tensor:
        """Forward pass with either specified or random operation."""
        if op_name is None:
            op_name = random.choice(self.op_names)
        if op_name == 'skip' and self.ops['skip'] is None:
            return torch.zeros_like(x)  # Return zero tensor if skip is not possible
        return self.ops[op_name](x)

class Supernet(nn.Module):
    """One-shot supernet containing all possible architectures."""
    def __init__(self, 
                 num_classes: int = 10,
                 num_layers: int = 8,
                 init_channels: int = 16):
        super().__init__()
        self.num_layers = num_layers
        C = init_channels
        
        # Initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU()
        )
        
        # Mixed layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Double channels at reduction layers
            if i in [num_layers//3, 2*num_layers//3]:
                C *= 2
                stride = 2
            else:
                stride = 1
            self.layers.append(MixedLayer(C, C, stride))
        
        # Global pooling and classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, num_classes)
        
        # Architecture sampling history
        self.sample_history = {}

    def forward(self, x: torch.Tensor, architecture: List[str] = None) -> torch.Tensor:
        """Forward pass with either specified or random architecture."""
        x = self.stem(x)
        
        if architecture is None:
            architecture = [None] * self.num_layers
        
        for layer, op_name in zip(self.layers, architecture):
            x = layer(x, op_name)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def sample_architecture(self) -> List[str]:
        """Sample random architecture."""
        architecture = []
        for layer in self.layers:
            op_name = random.choice(layer.op_names)
            architecture.append(op_name)
        return architecture

class OneShotNAS:
    """One-shot Neural Architecture Search with uniform sampling."""
    def __init__(self, 
                 supernet: Supernet,
                 num_architectures: int = 1000,
                 epochs: int = 50):
        self.supernet = supernet
        self.num_architectures = num_architectures
        self.epochs = epochs
        self.device = next(supernet.parameters()).device
        
        # Track best architectures
        self.best_architectures = []
        self.best_accuracies = []

    def train_supernet(self, 
                      train_loader: DataLoader,
                      optimizer: torch.optim.Optimizer):
        """Train supernet with random path sampling."""
        self.supernet.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Sample random architecture
            architecture = self.supernet.sample_architecture()
            
            optimizer.zero_grad()
            output = self.supernet(data, architecture)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Loss: {loss.item():.4f}')

    def evaluate_architecture(self, 
                            architecture: List[str],
                            valid_loader: DataLoader) -> float:
        """Evaluate specific architecture using supernet weights."""
        self.supernet.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.supernet(data, architecture)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total

    def search(self, valid_loader: DataLoader) -> Tuple[List[str], float]:
        """Search for best architecture by sampling and evaluation."""
        best_arch = None
        best_acc = 0
        
        for i in range(self.num_architectures):
            architecture = self.supernet.sample_architecture()
            accuracy = self.evaluate_architecture(architecture, valid_loader)
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_arch = architecture
                print(f"\nNew best architecture found!")
                print(f"Architecture: {best_arch}")
                print(f"Validation accuracy: {best_acc:.4f}")
        
        return best_arch, best_acc

def get_data_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 data loaders."""
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, valid_loader

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = get_data_loaders()
    
    # Create supernet
    supernet = Supernet().to(device)
    optimizer = torch.optim.SGD(
        supernet.parameters(),
        lr=0.025,
        momentum=0.9,
        weight_decay=3e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50)
    
    # Create NAS searcher
    nas = OneShotNAS(supernet)
    
    # Training loop
    print("Training supernet...")
    for epoch in range(nas.epochs):
        print(f"\nEpoch {epoch+1}/{nas.epochs}")
        nas.train_supernet(train_loader, optimizer)
        scheduler.step()
    
    # Architecture search
    print("\nSearching for best architecture...")
    best_architecture, best_accuracy = nas.search(valid_loader)
    
    print("\nSearch completed!")
    print(f"Best architecture: {best_architecture}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
