import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Dict, Tuple
import random
from scipy.stats import kendalltau

class SimpleCNN(nn.Module):
    """Simple CNN for architecture evaluation."""
    def __init__(self, num_classes=10):
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
            'maxpool3x3': nn.MaxPool2d(3, stride=1, padding=1),
            'avgpool3x3': nn.AvgPool2d(3, stride=1, padding=1),
            'skip': nn.Identity(),
            'none': nn.Identity()
        })
        
        self.classifier = nn.Linear(16, num_classes)
    
    def forward(self, x, architecture):
        x = self.stem(x)
        
        for op_name in architecture:
            x = self.ops[op_name](x)
            x = F.relu(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ZeroCostMetrics:
    """Computes zero-cost proxy metrics for architecture evaluation."""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def compute_grad_norm(self, 
                         architecture: List[str],
                         inputs: torch.Tensor,
                         targets: torch.Tensor) -> float:
        """Compute gradient norm at initialization."""
        self.model.zero_grad()
        outputs = self.model(inputs, architecture)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return np.sqrt(total_norm)
    
    def compute_snip_score(self,
                          architecture: List[str],
                          inputs: torch.Tensor,
                          targets: torch.Tensor) -> float:
        """Compute SNIP score (connection sensitivity)."""
        self.model.zero_grad()
        outputs = self.model(inputs, architecture)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        score = 0
        for name, p in self.model.named_parameters():
            if 'weight' in name and p.grad is not None:
                score += (p * p.grad).abs().sum().item()
        return score
    
    def compute_jacob_cov(self,
                         architecture: List[str],
                         inputs: torch.Tensor) -> float:
        """Compute Jacobian covariance score."""
        self.model.zero_grad()
        outputs = self.model(inputs, architecture)
        
        jacob = torch.zeros(outputs.shape[1], inputs.numel()).to(self.device)
        for i in range(outputs.shape[1]):
            if i > 0:
                self.model.zero_grad()
            outputs[0, i].backward(retain_graph=True)
            jacob[i] = inputs.grad.flatten()
        
        return torch.mean(torch.abs(torch.mm(jacob, jacob.t()))).item()
    
    def compute_all_metrics(self,
                          architecture: List[str],
                          inputs: torch.Tensor,
                          targets: torch.Tensor) -> Dict[str, float]:
        """Compute all zero-cost metrics."""
        metrics = {
            'grad_norm': self.compute_grad_norm(architecture, inputs, targets),
            'snip': self.compute_snip_score(architecture, inputs, targets),
            'jacob_cov': self.compute_jacob_cov(architecture, inputs)
        }
        return metrics

class ZeroCostNAS:
    """Neural Architecture Search using zero-cost proxies."""
    def __init__(self,
                 model: SimpleCNN,
                 num_architectures: int = 1000):
        self.model = model
        self.device = next(model.parameters()).device
        self.metrics = ZeroCostMetrics(model, self.device)
        self.num_architectures = num_architectures
        
        # Available operations
        self.operations = [
            'conv3x3', 'conv5x5', 'maxpool3x3',
            'avgpool3x3', 'skip', 'none'
        ]
    
    def generate_random_architecture(self, length: int = 8) -> List[str]:
        """Generate random architecture."""
        return [random.choice(self.operations) for _ in range(length)]
    
    def normalize_scores(self, scores: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Normalize scores to [0, 1] range."""
        normalized = {}
        for metric, values in scores.items():
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                normalized[metric] = [(v - min_val) / (max_val - min_val) 
                                   for v in values]
            else:
                normalized[metric] = [0.5 for _ in values]
        return normalized
    
    def compute_final_score(self, 
                          normalized_scores: Dict[str, List[float]],
                          weights: Dict[str, float] = None) -> List[float]:
        """Compute weighted average of normalized scores."""
        if weights is None:
            weights = {
                'grad_norm': 1.0,
                'snip': 1.0,
                'jacob_cov': 1.0
            }
        
        total_weight = sum(weights.values())
        final_scores = []
        
        for i in range(len(next(iter(normalized_scores.values())))):
            score = sum(weights[metric] * normalized_scores[metric][i]
                       for metric in weights)
            final_scores.append(score / total_weight)
        
        return final_scores
    
    def search(self,
              train_loader: DataLoader,
              valid_loader: DataLoader = None) -> Tuple[List[str], Dict[str, float]]:
        """Perform architecture search using zero-cost proxies."""
        print("Generating and evaluating architectures...")
        
        # Get batch of data for metrics computation
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Generate and evaluate architectures
        architectures = []
        metric_scores = {
            'grad_norm': [],
            'snip': [],
            'jacob_cov': []
        }
        
        for i in range(self.num_architectures):
            if i % 100 == 0:
                print(f"Evaluated {i}/{self.num_architectures} architectures")
            
            arch = self.generate_random_architecture()
            metrics = self.metrics.compute_all_metrics(arch, inputs, targets)
            
            architectures.append(arch)
            for metric, score in metrics.items():
                metric_scores[metric].append(score)
        
        # Normalize scores and compute final ranking
        normalized_scores = self.normalize_scores(metric_scores)
        final_scores = self.compute_final_score(normalized_scores)
        
        # Select best architecture
        best_idx = np.argmax(final_scores)
        best_architecture = architectures[best_idx]
        best_metrics = {
            metric: scores[best_idx] 
            for metric, scores in metric_scores.items()
        }
        
        return best_architecture, best_metrics

def evaluate_architecture(
    architecture: List[str],
    model: SimpleCNN,
    valid_loader: DataLoader
) -> float:
    """Evaluate architecture accuracy on validation set."""
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, architecture)
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
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64)
    
    # Create model and searcher
    model = SimpleCNN().to(device)
    searcher = ZeroCostNAS(model)
    
    # Perform search
    print("\nStarting Zero-Cost NAS...")
    best_architecture, best_metrics = searcher.search(train_loader)
    
    print("\nSearch completed!")
    print(f"Best architecture: {best_architecture}")
    print("\nProxy metrics for best architecture:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Evaluate best architecture
    accuracy = evaluate_architecture(best_architecture, model, valid_loader)
    print(f"\nValidation accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
