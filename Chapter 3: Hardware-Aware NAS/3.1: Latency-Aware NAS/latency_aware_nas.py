import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Dict, Tuple
import time
import json
import os
from pathlib import Path

class LatencyEstimator:
    """Estimates latency of operations on target hardware."""
    def __init__(self, device_type: str = "cpu"):
        self.device_type = device_type
        self.cache_file = f"latency_cache_{device_type}.json"
        self.latency_cache = self._load_cache()
        
        # Base latency measurements (in ms) for different operations
        # These would ideally be measured on actual hardware
        self.base_latencies = {
            'conv3x3': 1.0,
            'conv5x5': 2.5,
            'conv7x7': 4.0,
            'maxpool3x3': 0.5,
            'avgpool3x3': 0.5,
            'skip': 0.1
        }
    
    def _load_cache(self) -> Dict:
        """Load cached latency measurements."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save latency measurements to cache."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.latency_cache, f)
    
    def measure_latency(self, 
                       model: nn.Module,
                       input_size: Tuple[int, int, int, int],
                       architecture: List[str]) -> float:
        """Measure actual latency on hardware."""
        device = next(model.parameters()).device
        x = torch.randn(*input_size).to(device)
        
        # Warm-up
        for _ in range(10):
            model(x, architecture)
        
        # Measure
        start_time = time.time()
        iterations = 100
        for _ in range(iterations):
            model(x, architecture)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        return (time.time() - start_time) * 1000 / iterations  # Convert to ms
    
    def estimate_latency(self, architecture: List[str]) -> float:
        """Estimate latency using cached measurements or base latencies."""
        arch_key = '_'.join(architecture)
        
        if arch_key in self.latency_cache:
            return self.latency_cache[arch_key]
        
        # Estimate using base latencies
        total_latency = sum(self.base_latencies[op] for op in architecture)
        return total_latency

class LatencyAwareNAS:
    """Neural Architecture Search with latency constraints."""
    def __init__(self,
                 model: nn.Module,
                 latency_constraint: float,
                 device_type: str = "cpu"):
        self.model = model
        self.latency_constraint = latency_constraint
        self.latency_estimator = LatencyEstimator(device_type)
        self.operations = [
            'conv3x3', 'conv5x5', 'conv7x7',
            'maxpool3x3', 'avgpool3x3', 'skip'
        ]
        
        # History for tracking search progress
        self.history = []
    
    def generate_random_architecture(self, length: int = 8) -> List[str]:
        """Generate random architecture."""
        return [np.random.choice(self.operations) for _ in range(length)]
    
    def evaluate_architecture(self,
                            architecture: List[str],
                            valid_loader: DataLoader,
                            device: torch.device) -> Tuple[float, float]:
        """Evaluate architecture for both accuracy and latency."""
        # Evaluate accuracy
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs, architecture)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        accuracy = correct / total
        
        # Estimate latency
        latency = self.latency_estimator.estimate_latency(architecture)
        
        return accuracy, latency
    
    def search(self,
               train_loader: DataLoader,
               valid_loader: DataLoader,
               device: torch.device,
               num_iterations: int = 1000) -> Tuple[List[str], float, float]:
        """Perform latency-constrained architecture search."""
        best_architecture = None
        best_accuracy = 0
        best_latency = float('inf')
        
        for i in range(num_iterations):
            if i % 100 == 0:
                print(f"Search iteration {i}/{num_iterations}")
            
            # Generate candidate architecture
            architecture = self.generate_random_architecture()
            
            # Evaluate architecture
            accuracy, latency = self.evaluate_architecture(
                architecture, valid_loader, device)
            
            # Update best architecture if it meets latency constraint
            if latency <= self.latency_constraint and accuracy > best_accuracy:
                best_architecture = architecture
                best_accuracy = accuracy
                best_latency = latency
                
                print(f"\nNew best architecture found!")
                print(f"Accuracy: {best_accuracy:.4f}")
                print(f"Latency: {best_latency:.2f}ms")
                print(f"Architecture: {best_architecture}")
            
            # Store in history
            self.history.append({
                'architecture': architecture,
                'accuracy': accuracy,
                'latency': latency,
                'iteration': i
            })
        
        return best_architecture, best_accuracy, best_latency
    
    def plot_search_trajectory(self):
        """Plot accuracy vs latency for all evaluated architectures."""
        import matplotlib.pyplot as plt
        
        accuracies = [h['accuracy'] for h in self.history]
        latencies = [h['latency'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(latencies, accuracies, alpha=0.5)
        plt.axvline(x=self.latency_constraint, color='r', linestyle='--',
                   label='Latency Constraint')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Latency Trade-off')
        plt.legend()
        plt.show()

class LatencyAwareCNN(nn.Module):
    """CNN with latency-aware design."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Available operations
        self.ops = nn.ModuleDict({
            'conv3x3': nn.Conv2d(16, 16, 3, padding=1),
            'conv5x5': nn.Conv2d(16, 16, 5, padding=2),
            'conv7x7': nn.Conv2d(16, 16, 7, padding=3),
            'maxpool3x3': nn.MaxPool2d(3, stride=1, padding=1),
            'avgpool3x3': nn.AvgPool2d(3, stride=1, padding=1),
            'skip': nn.Identity()
        })
        
        self.classifier = nn.Linear(16, num_classes)
    
    def forward(self, x: torch.Tensor, architecture: List[str]) -> torch.Tensor:
        x = self.stem(x)
        
        for op_name in architecture:
            x = self.ops[op_name](x)
            x = F.relu(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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
    
    # Create model and searcher
    model = LatencyAwareCNN().to(device)
    searcher = LatencyAwareNAS(
        model=model,
        latency_constraint=10.0,  # 10ms latency constraint
        device_type=device.type
    )
    
    # Perform search
    print("Starting latency-aware architecture search...")
    best_arch, best_acc, best_lat = searcher.search(
        train_loader, valid_loader, device)
    
    print("\nSearch completed!")
    print(f"Best architecture: {best_arch}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Latency: {best_lat:.2f}ms")
    
    # Plot results
    searcher.plot_search_trajectory()

if __name__ == "__main__":
    main()
