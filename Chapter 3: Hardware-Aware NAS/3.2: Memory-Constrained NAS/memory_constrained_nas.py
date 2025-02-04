import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Dict, Tuple
import math

class MemoryEstimator:
    """Estimates memory requirements for neural architectures."""
    def __init__(self, input_size: Tuple[int, int, int]):
        self.input_size = input_size  # (C, H, W)
        
        # Memory requirements in bytes for different data types
        self.dtype_memory = {
            torch.float32: 4,
            torch.float16: 2,
            torch.int8: 1
        }
        
        # Base memory for different operations (parameters)
        self.op_param_memory = {
            'conv3x3': lambda c_in, c_out: 9 * c_in * c_out,
            'conv5x5': lambda c_in, c_out: 25 * c_in * c_out,
            'conv7x7': lambda c_in, c_out: 49 * c_in * c_out,
            'maxpool3x3': lambda c_in, c_out: 0,
            'avgpool3x3': lambda c_in, c_out: 0,
            'skip': lambda c_in, c_out: 0
        }
    
    def estimate_parameter_memory(self, 
                                architecture: List[str],
                                channels: List[int],
                                dtype: torch.dtype = torch.float32) -> int:
        """Estimate memory needed for parameters."""
        total_params = 0
        bytes_per_param = self.dtype_memory[dtype]
        
        for i, op_name in enumerate(architecture):
            c_in = channels[i]
            c_out = channels[i + 1]
            total_params += self.op_param_memory[op_name](c_in, c_out)
        
        return total_params * bytes_per_param
    
    def estimate_activation_memory(self,
                                 architecture: List[str],
                                 channels: List[int],
                                 batch_size: int,
                                 dtype: torch.dtype = torch.float32) -> int:
        """Estimate memory needed for activations."""
        H, W = self.input_size[1:]
        bytes_per_element = self.dtype_memory[dtype]
        total_activation_memory = 0
        
        # Input activation memory
        current_size = batch_size * channels[0] * H * W * bytes_per_element
        total_activation_memory = current_size
        
        for i, op_name in enumerate(architecture):
            # Update activation size based on operation
            if 'conv' in op_name or 'pool' in op_name:
                current_size = batch_size * channels[i + 1] * H * W * bytes_per_element
            total_activation_memory = max(total_activation_memory, current_size)
        
        return total_activation_memory
    
    def estimate_total_memory(self,
                            architecture: List[str],
                            channels: List[int],
                            batch_size: int,
                            dtype: torch.dtype = torch.float32) -> Dict[str, int]:
        """Estimate total memory requirements."""
        param_memory = self.estimate_parameter_memory(architecture, channels, dtype)
        activation_memory = self.estimate_activation_memory(
            architecture, channels, batch_size, dtype)
        
        return {
            'parameter_memory': param_memory,
            'activation_memory': activation_memory,
            'total_memory': param_memory + activation_memory
        }

class MemoryConstrainedNAS:
    """Neural Architecture Search with memory constraints."""
    def __init__(self,
                 input_size: Tuple[int, int, int],
                 memory_constraint: int,  # in bytes
                 min_channels: int = 16,
                 max_channels: int = 128):
        self.memory_estimator = MemoryEstimator(input_size)
        self.memory_constraint = memory_constraint
        self.min_channels = min_channels
        self.max_channels = max_channels
        
        self.operations = [
            'conv3x3', 'conv5x5', 'conv7x7',
            'maxpool3x3', 'avgpool3x3', 'skip'
        ]
    
    def generate_random_architecture(self,
                                   num_layers: int = 8) -> Tuple[List[str], List[int]]:
        """Generate random architecture with channel configurations."""
        architecture = []
        channels = [self.min_channels]  # Input channels
        
        for _ in range(num_layers):
            # Sample operation
            op = np.random.choice(self.operations)
            architecture.append(op)
            
            # Sample output channels
            out_channels = np.random.randint(self.min_channels, self.max_channels + 1)
            channels.append(out_channels)
        
        return architecture, channels
    
    def mutate_architecture(self,
                          architecture: List[str],
                          channels: List[int]) -> Tuple[List[str], List[int]]:
        """Mutate architecture while considering memory constraints."""
        new_architecture = architecture.copy()
        new_channels = channels.copy()
        
        # Randomly choose mutation type
        mutation_type = np.random.choice(['operation', 'channels'])
        
        if mutation_type == 'operation':
            # Mutate random operation
            idx = np.random.randint(len(architecture))
            new_architecture[idx] = np.random.choice(self.operations)
        else:
            # Mutate random channel size
            idx = np.random.randint(1, len(channels))
            delta = np.random.randint(-16, 17)  # Change by up to 16 channels
            new_channels[idx] = np.clip(
                channels[idx] + delta, self.min_channels, self.max_channels)
        
        return new_architecture, new_channels

class MemoryEfficientNet(nn.Module):
    """Neural network with memory-efficient design."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Dynamic operations
        self.ops = {
            'conv3x3': lambda c_in, c_out: nn.Conv2d(c_in, c_out, 3, padding=1),
            'conv5x5': lambda c_in, c_out: nn.Conv2d(c_in, c_out, 5, padding=2),
            'conv7x7': lambda c_in, c_out: nn.Conv2d(c_in, c_out, 7, padding=3),
            'maxpool3x3': lambda c_in, c_out: nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
            ),
            'avgpool3x3': lambda c_in, c_out: nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
            ),
            'skip': lambda c_in, c_out: nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        }
    
    def create_layer(self, op_name: str, c_in: int, c_out: int) -> nn.Module:
        """Create layer dynamically based on operation name and channels."""
        return nn.Sequential(
            self.ops[op_name](c_in, c_out),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )
    
    def forward(self, 
                x: torch.Tensor,
                architecture: List[str],
                channels: List[int]) -> torch.Tensor:
        x = self.stem(x)
        
        # Apply operations with specified channels
        for i, op_name in enumerate(architecture):
            layer = self.create_layer(op_name, channels[i], channels[i + 1])
            x = layer(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return nn.Linear(channels[-1], 10).to(x.device)(x)

def search_memory_constrained(
    model: MemoryEfficientNet,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    memory_constraint: int,
    device: torch.device,
    num_iterations: int = 1000
) -> Tuple[List[str], List[int], float]:
    """Perform memory-constrained architecture search."""
    nas = MemoryConstrainedNAS(
        input_size=(3, 32, 32),
        memory_constraint=memory_constraint
    )
    
    best_architecture = None
    best_channels = None
    best_accuracy = 0
    best_memory = float('inf')
    
    for i in range(num_iterations):
        if i % 100 == 0:
            print(f"Search iteration {i}/{num_iterations}")
        
        # Generate or mutate architecture
        if best_architecture is None:
            architecture, channels = nas.generate_random_architecture()
        else:
            architecture, channels = nas.mutate_architecture(
                best_architecture, best_channels)
        
        # Check memory constraints
        memory_usage = nas.memory_estimator.estimate_total_memory(
            architecture, channels, batch_size=train_loader.batch_size)
        
        if memory_usage['total_memory'] <= memory_constraint:
            # Evaluate architecture
            accuracy = evaluate_architecture(
                model, architecture, channels, valid_loader, device)
            
            if accuracy > best_accuracy:
                best_architecture = architecture
                best_channels = channels
                best_accuracy = accuracy
                best_memory = memory_usage['total_memory']
                
                print(f"\nNew best architecture found!")
                print(f"Accuracy: {best_accuracy:.4f}")
                print(f"Memory usage: {best_memory/1e6:.2f}MB")
    
    return best_architecture, best_channels, best_accuracy

def evaluate_architecture(
    model: MemoryEfficientNet,
    architecture: List[str],
    channels: List[int],
    valid_loader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate architecture accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, architecture, channels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return correct / total

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=128)
    
    # Create model
    model = MemoryEfficientNet().to(device)
    
    # Set memory constraint (100MB)
    memory_constraint = 100 * 1024 * 1024  # bytes
    
    # Perform search
    print("Starting memory-constrained architecture search...")
    best_arch, best_channels, best_acc = search_memory_constrained(
        model, train_loader, valid_loader, memory_constraint, device)
    
    print("\nSearch completed!")
    print(f"Best architecture: {best_arch}")
    print(f"Channel configuration: {best_channels}")
    print(f"Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
