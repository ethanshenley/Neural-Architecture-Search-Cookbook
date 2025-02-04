import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Dict, Tuple
import random

class ElasticLinear(nn.Module):
    """Linear layer with elastic width."""
    def __init__(self, max_in_features: int, max_out_features: int):
        super().__init__()
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.weight = nn.Parameter(torch.Tensor(max_out_features, max_in_features))
        self.bias = nn.Parameter(torch.Tensor(max_out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor, out_features: int = None) -> torch.Tensor:
        """Forward pass with dynamic output features."""
        out_features = out_features or self.max_out_features
        weight = self.weight[:out_features, :x.size(1)]
        bias = self.bias[:out_features]
        return F.linear(x, weight, bias)

class ElasticConv2d(nn.Module):
    """Convolutional layer with elastic width and kernel size."""
    def __init__(self, 
                 max_in_channels: int,
                 max_out_channels: int,
                 kernel_sizes: List[int]):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_sizes = sorted(kernel_sizes)
        self.max_kernel_size = max(kernel_sizes)
        
        # Create weight for largest kernel size
        self.weight = nn.Parameter(
            torch.Tensor(max_out_channels, max_in_channels, 
                        self.max_kernel_size, self.max_kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(max_out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, 
                x: torch.Tensor,
                out_channels: int = None,
                kernel_size: int = None) -> torch.Tensor:
        """Forward pass with dynamic channels and kernel size."""
        out_channels = out_channels or self.max_out_channels
        kernel_size = kernel_size or self.max_kernel_size
        
        if kernel_size == self.max_kernel_size:
            weight = self.weight[:out_channels, :x.size(1), :, :]
        else:
            # Center crop weight for smaller kernel sizes
            diff = self.max_kernel_size - kernel_size
            start = diff // 2
            end = start + kernel_size
            weight = self.weight[:out_channels, :x.size(1), start:end, start:end]
        
        bias = self.bias[:out_channels]
        padding = kernel_size // 2
        return F.conv2d(x, weight, bias, padding=padding)

class ElasticBlock(nn.Module):
    """Basic block with elastic operations."""
    def __init__(self,
                 max_in_channels: int,
                 max_out_channels: int,
                 kernel_sizes: List[int]):
        super().__init__()
        self.conv = ElasticConv2d(max_in_channels, max_out_channels, kernel_sizes)
        self.bn = nn.BatchNorm2d(max_out_channels)
    
    def forward(self,
                x: torch.Tensor,
                out_channels: int = None,
                kernel_size: int = None) -> torch.Tensor:
        x = self.conv(x, out_channels, kernel_size)
        if out_channels is not None:
            x = self.bn(x)[:, :out_channels]
        else:
            x = self.bn(x)
        return F.relu(x)

class OnceForAllNetwork(nn.Module):
    """Once-for-all network supporting multiple sub-networks."""
    def __init__(self,
                 num_classes: int = 10,
                 min_channels: int = 16,
                 max_channels: int = 64,
                 min_layers: int = 4,
                 max_layers: int = 8,
                 kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.kernel_sizes = kernel_sizes
        
        # Initial stem convolution
        self.stem = ElasticBlock(3, max_channels, kernel_sizes)
        
        # Elastic layers
        self.layers = nn.ModuleList([
            ElasticBlock(max_channels, max_channels, kernel_sizes)
            for _ in range(max_layers)
        ])
        
        # Classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = ElasticLinear(max_channels, num_classes)
    
    def forward(self, x: torch.Tensor, config: Dict = None) -> torch.Tensor:
        """Forward pass with dynamic architecture configuration."""
        if config is None:
            config = self.sample_config()
        
        # Apply stem with initial width
        x = self.stem(x, config['widths'][0], config['kernel_sizes'][0])
        
        # Apply elastic layers
        for i in range(config['depth']):
            x = self.layers[i](x, config['widths'][i+1], config['kernel_sizes'][i+1])
        
        # Classification
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def sample_config(self) -> Dict:
        """Sample random sub-network configuration."""
        depth = random.randint(self.min_layers, self.max_layers)
        widths = [random.randint(self.min_channels, self.max_channels)
                 for _ in range(depth + 1)]  # +1 for stem
        kernel_sizes = [random.choice(self.kernel_sizes)
                       for _ in range(depth + 1)]  # +1 for stem
        return {
            'depth': depth,
            'widths': widths,
            'kernel_sizes': kernel_sizes
        }

class ProgressiveShrinking:
    """Progressive shrinking training strategy."""
    def __init__(self,
                 model: OnceForAllNetwork,
                 teacher_model: nn.Module = None):
        self.model = model
        self.teacher_model = teacher_model
    
    def train_step(self,
                  inputs: torch.Tensor,
                  targets: torch.Tensor,
                  optimizer: torch.optim.Optimizer,
                  config: Dict = None) -> float:
        """Single training step with optional knowledge distillation."""
        optimizer.zero_grad()
        
        # Forward pass with sampled or provided config
        outputs = self.model(inputs, config)
        
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(outputs, targets)
        
        # Add knowledge distillation if teacher is available
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
            kd_loss = F.kl_div(
                F.log_softmax(outputs / 2.0, dim=1),
                F.softmax(teacher_outputs / 2.0, dim=1),
                reduction='batchmean'
            ) * (2.0 ** 2)
            loss = 0.5 * (ce_loss + kd_loss)
        else:
            loss = ce_loss
        
        loss.backward()
        optimizer.step()
        return loss.item()

def train_once_for_all(
    model: OnceForAllNetwork,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int = 100
):
    """Train once-for-all network with progressive shrinking."""
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.025,
        momentum=0.9,
        weight_decay=3e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs)
    
    # Create largest model as teacher
    teacher_config = {
        'depth': model.max_layers,
        'widths': [model.max_channels] * (model.max_layers + 1),
        'kernel_sizes': [max(model.kernel_sizes)] * (model.max_layers + 1)
    }
    
    trainer = ProgressiveShrinking(model)
    
    # Training stages
    stages = [
        {'epochs': epochs // 4, 'config': teacher_config},  # Largest model
        {'epochs': epochs // 4, 'config': None},  # Random kernel sizes
        {'epochs': epochs // 4, 'config': None},  # Random widths
        {'epochs': epochs // 4, 'config': None}   # Random depths
    ]
    
    for stage_idx, stage in enumerate(stages):
        print(f"\nStage {stage_idx + 1}/{len(stages)}")
        
        for epoch in range(stage['epochs']):
            model.train()
            total_loss = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                loss = trainer.train_step(inputs, targets, optimizer, stage['config'])
                total_loss += loss
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss:.4f}")
            
            scheduler.step()
            
            # Validation
            if epoch % 5 == 0:
                validate_once_for_all(model, valid_loader, device)

def validate_once_for_all(
    model: OnceForAllNetwork,
    valid_loader: DataLoader,
    device: torch.device,
    num_samples: int = 10
):
    """Validate once-for-all network with random architectures."""
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            config = model.sample_config()
            correct = 0
            total = 0
            
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, config)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
            
            accuracy = correct / total
            accuracies.append(accuracy)
            print(f"Config: {config}")
            print(f"Accuracy: {accuracy:.4f}")
    
    print(f"\nMean accuracy: {np.mean(accuracies):.4f}")
    print(f"Std accuracy: {np.std(accuracies):.4f}")

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
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
    
    train_loader = DataLoader(train_data, batch_size=96, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=96)
    
    # Create model
    model = OnceForAllNetwork().to(device)
    
    # Train model
    print("Starting Once-for-All training...")
    train_once_for_all(model, train_loader, valid_loader, device)
    
    # Final validation
    print("\nFinal validation with different architectures:")
    validate_once_for_all(model, valid_loader, device)

if __name__ == "__main__":
    main()
