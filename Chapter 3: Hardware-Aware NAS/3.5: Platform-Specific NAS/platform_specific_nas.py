import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
import os

class Platform(Enum):
    """Supported hardware platforms."""
    MOBILE_ARM = "mobile_arm"
    EDGE_TPU = "edge_tpu"
    FPGA = "fpga"
    CPU_X86 = "cpu_x86"

class PlatformProfile:
    """Hardware platform specifications and constraints."""
    def __init__(self, platform: Platform):
        self.platform = platform
        
        # Platform-specific constraints
        self.constraints = {
            Platform.MOBILE_ARM: {
                'max_memory': 512 * 1024 * 1024,  # 512MB
                'max_compute': 2 * 1024 * 1024 * 1024,  # 2 GFLOPS
                'max_power': 2.5,  # 2.5W
                'supported_ops': ['conv3x3', 'conv5x5', 'maxpool3x3', 'avgpool3x3', 'skip'],
                'preferred_ops': ['conv3x3', 'maxpool3x3'],  # Operations optimized for platform
                'min_channel_align': 8,  # Channel alignment requirement
                'max_channels': 256
            },
            Platform.EDGE_TPU: {
                'max_memory': 8 * 1024 * 1024,  # 8MB
                'max_compute': 4 * 1024 * 1024 * 1024,  # 4 TOPS
                'max_power': 2.0,  # 2W
                'supported_ops': ['conv3x3', 'conv5x5', 'avgpool3x3', 'skip'],
                'preferred_ops': ['conv3x3'],  # TPU-optimized operations
                'min_channel_align': 128,
                'max_channels': 512
            },
            Platform.FPGA: {
                'max_memory': 256 * 1024 * 1024,  # 256MB
                'max_compute': 1 * 1024 * 1024 * 1024,  # 1 TOPS
                'max_power': 5.0,  # 5W
                'supported_ops': ['conv3x3', 'conv5x5', 'maxpool3x3'],
                'preferred_ops': ['conv3x3', 'conv5x5'],  # FPGA-optimized operations
                'min_channel_align': 16,
                'max_channels': 1024
            },
            Platform.CPU_X86: {
                'max_memory': 4 * 1024 * 1024 * 1024,  # 4GB
                'max_compute': 8 * 1024 * 1024 * 1024,  # 8 GFLOPS
                'max_power': 15.0,  # 15W
                'supported_ops': ['conv3x3', 'conv5x5', 'conv7x7', 'maxpool3x3', 'avgpool3x3', 'skip'],
                'preferred_ops': ['conv3x3', 'conv5x5'],  # CPU-optimized operations
                'min_channel_align': 4,
                'max_channels': 2048
            }
        }
        
        # Load platform-specific performance data
        self.perf_data = self._load_performance_data()
    
    def _load_performance_data(self) -> Dict:
        """Load cached performance measurements for the platform."""
        cache_file = f"perf_cache_{self.platform.value}.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def align_channels(self, channels: int) -> int:
        """Align channel count to platform requirements."""
        alignment = self.constraints[self.platform]['min_channel_align']
        return ((channels + alignment - 1) // alignment) * alignment
    
    def is_supported_op(self, op_name: str) -> bool:
        """Check if operation is supported on platform."""
        return op_name in self.constraints[self.platform]['supported_ops']
    
    def is_preferred_op(self, op_name: str) -> bool:
        """Check if operation is preferred/optimized for platform."""
        return op_name in self.constraints[self.platform]['preferred_ops']
    
    def estimate_latency(self, op_name: str, input_size: int, output_size: int) -> float:
        """Estimate operation latency on platform."""
        key = f"{op_name}_{input_size}_{output_size}"
        return self.perf_data.get(key, float('inf'))
    
    def check_constraints(self,
                         architecture: List[str],
                         channels: List[int]) -> Tuple[bool, Dict]:
        """Check if architecture meets platform constraints."""
        total_memory = sum(c * c * 4 for c in channels)  # Rough estimate
        total_compute = sum(
            c_in * c_out * 9 for c_in, c_out in zip(channels[:-1], channels[1:]))
        max_channels = max(channels)
        
        constraints = self.constraints[self.platform]
        
        return (
            total_memory <= constraints['max_memory'] and
            total_compute <= constraints['max_compute'] and
            max_channels <= constraints['max_channels'] and
            all(self.is_supported_op(op) for op in architecture)
        ), {
            'memory_usage': total_memory,
            'compute_usage': total_compute,
            'max_channels': max_channels
        }

class PlatformSpecificNAS:
    """Neural Architecture Search optimized for specific platforms."""
    def __init__(self,
                 platform: Platform,
                 min_channels: int = 16):
        self.platform = platform
        self.profile = PlatformProfile(platform)
        self.min_channels = self.profile.align_channels(min_channels)
        
        # Get supported operations for platform
        self.operations = self.profile.constraints[platform]['supported_ops']
    
    def generate_random_architecture(self,
                                   num_layers: int = 8) -> Tuple[List[str], List[int]]:
        """Generate random architecture suitable for platform."""
        architecture = []
        channels = [self.min_channels]
        
        for _ in range(num_layers):
            # Prefer platform-optimized operations
            if np.random.random() < 0.7:  # 70% chance to use preferred ops
                op = np.random.choice(
                    self.profile.constraints[self.platform]['preferred_ops'])
            else:
                op = np.random.choice(self.operations)
            architecture.append(op)
            
            # Generate aligned channel count
            max_channels = self.profile.constraints[self.platform]['max_channels']
            channels.append(self.profile.align_channels(
                np.random.randint(self.min_channels, max_channels + 1)))
        
        return architecture, channels
    
    def mutate_architecture(self,
                          architecture: List[str],
                          channels: List[int]) -> Tuple[List[str], List[int]]:
        """Mutate architecture while respecting platform constraints."""
        new_architecture = architecture.copy()
        new_channels = channels.copy()
        
        # Randomly choose mutation type
        mutation_type = np.random.choice(['operation', 'channels'])
        
        if mutation_type == 'operation':
            # Mutate random operation
            idx = np.random.randint(len(architecture))
            if np.random.random() < 0.7:  # Prefer platform-optimized operations
                new_architecture[idx] = np.random.choice(
                    self.profile.constraints[self.platform]['preferred_ops'])
            else:
                new_architecture[idx] = np.random.choice(self.operations)
        else:
            # Mutate random channel size
            idx = np.random.randint(1, len(channels))
            max_channels = self.profile.constraints[self.platform]['max_channels']
            new_channels[idx] = self.profile.align_channels(
                np.random.randint(self.min_channels, max_channels + 1))
        
        return new_architecture, new_channels

class PlatformSpecificNet(nn.Module):
    """Neural network optimized for specific platform."""
    def __init__(self, platform: Platform, num_classes: int = 10):
        super().__init__()
        self.platform = platform
        self.profile = PlatformProfile(platform)
        
        # Platform-optimized stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Platform-specific operations
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
        """Create platform-optimized layer."""
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
        
        for i, op_name in enumerate(architecture):
            layer = self.create_layer(op_name, channels[i], channels[i + 1])
            x = layer(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return nn.Linear(channels[-1], 10).to(x.device)(x)

def search_platform_specific(
    model: PlatformSpecificNet,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    platform: Platform,
    device: torch.device,
    num_iterations: int = 1000
) -> Tuple[List[str], List[int], float]:
    """Perform platform-specific architecture search."""
    nas = PlatformSpecificNAS(platform=platform)
    
    best_architecture = None
    best_channels = None
    best_accuracy = 0
    
    for i in range(num_iterations):
        if i % 100 == 0:
            print(f"Search iteration {i}/{num_iterations}")
        
        # Generate or mutate architecture
        if best_architecture is None:
            architecture, channels = nas.generate_random_architecture()
        else:
            architecture, channels = nas.mutate_architecture(
                best_architecture, best_channels)
        
        # Check platform constraints
        meets_constraints, stats = nas.profile.check_constraints(
            architecture, channels)
        
        if meets_constraints:
            # Evaluate architecture
            accuracy = evaluate_architecture(
                model, architecture, channels, valid_loader, device)
            
            if accuracy > best_accuracy:
                best_architecture = architecture
                best_channels = channels
                best_accuracy = accuracy
                
                print(f"\nNew best architecture found!")
                print(f"Accuracy: {best_accuracy:.4f}")
                print("Resource usage:")
                for k, v in stats.items():
                    print(f"  {k}: {v:,}")
    
    return best_architecture, best_channels, best_accuracy

def evaluate_architecture(
    model: PlatformSpecificNet,
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
    
    # Select target platform
    platform = Platform.MOBILE_ARM  # Can be changed to other platforms
    
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
    model = PlatformSpecificNet(platform=platform).to(device)
    
    # Perform search
    print(f"Starting platform-specific architecture search for {platform.value}...")
    best_arch, best_channels, best_acc = search_platform_specific(
        model, train_loader, valid_loader, platform, device)
    
    print("\nSearch completed!")
    print(f"Best architecture: {best_arch}")
    print(f"Channel configuration: {best_channels}")
    print(f"Accuracy: {best_acc:.4f}")
    
    # Check final constraints
    meets_constraints, stats = PlatformProfile(platform).check_constraints(
        best_arch, best_channels)
    print("\nFinal resource usage:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")

if __name__ == "__main__":
    main()
