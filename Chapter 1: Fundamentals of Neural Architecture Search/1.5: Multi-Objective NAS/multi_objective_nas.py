import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

class LatencyEstimator:
    """Estimates latency of operations on target hardware."""
    def __init__(self):
        # Simulated latency measurements (ms) for different operations
        self.op_latency = {
            'conv3x3': 1.0,
            'conv5x5': 2.5,
            'conv7x7': 4.0,
            'maxpool3x3': 0.3,
            'avgpool3x3': 0.2,
            'skip_connect': 0.1
        }
    
    def estimate_latency(self, architecture: List[str]) -> float:
        """Estimate total latency of an architecture."""
        return sum(self.op_latency[op] for op in architecture)

class ResourceEstimator:
    """Estimates computational resources (FLOPs, parameters)."""
    def __init__(self, input_shape=(3, 32, 32)):
        self.input_shape = input_shape
        
        # FLOPs for different operations (millions)
        self.op_flops = {
            'conv3x3': 0.15,
            'conv5x5': 0.4,
            'conv7x7': 0.8,
            'maxpool3x3': 0.01,
            'avgpool3x3': 0.01,
            'skip_connect': 0.0
        }
        
        # Parameters for different operations (thousands)
        self.op_params = {
            'conv3x3': 9,
            'conv5x5': 25,
            'conv7x7': 49,
            'maxpool3x3': 0,
            'avgpool3x3': 0,
            'skip_connect': 0
        }
    
    def estimate_resources(self, architecture: List[str]) -> Tuple[float, float]:
        """Estimate FLOPs and parameters of an architecture."""
        total_flops = sum(self.op_flops[op] for op in architecture)
        total_params = sum(self.op_params[op] for op in architecture)
        return total_flops, total_params

class MultiObjectiveNAS:
    """Multi-objective Neural Architecture Search."""
    def __init__(self, 
                 num_layers: int = 8,
                 population_size: int = 100,
                 num_generations: int = 50):
        self.num_layers = num_layers
        self.population_size = population_size
        self.num_generations = num_generations
        
        self.operations = [
            'conv3x3', 'conv5x5', 'conv7x7',
            'maxpool3x3', 'avgpool3x3', 'skip_connect'
        ]
        
        self.latency_estimator = LatencyEstimator()
        self.resource_estimator = ResourceEstimator()
        
        # Initialize population
        self.population = self._initialize_population()
        self.objectives = {}  # Cache for objective values
        
    def _initialize_population(self) -> List[List[str]]:
        """Initialize random population of architectures."""
        population = []
        for _ in range(self.population_size):
            architecture = [
                np.random.choice(self.operations)
                for _ in range(self.num_layers)
            ]
            population.append(architecture)
        return population
    
    def evaluate_architecture(self, 
                            architecture: List[str],
                            model: nn.Module,
                            valid_loader: DataLoader,
                            device: torch.device) -> Dict[str, float]:
        """Evaluate an architecture for all objectives."""
        if str(architecture) in self.objectives:
            return self.objectives[str(architecture)]
        
        # 1. Accuracy
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
        accuracy = correct / total
        
        # 2. Latency
        latency = self.latency_estimator.estimate_latency(architecture)
        
        # 3. Resource usage
        flops, params = self.resource_estimator.estimate_resources(architecture)
        
        objectives = {
            'accuracy': accuracy,
            'latency': -latency,  # Negative because we want to minimize
            'flops': -flops,      # Negative because we want to minimize
            'params': -params     # Negative because we want to minimize
        }
        
        self.objectives[str(architecture)] = objectives
        return objectives
    
    def is_dominated(self, obj1: Dict[str, float], obj2: Dict[str, float]) -> bool:
        """Check if obj1 is dominated by obj2."""
        worse_equal = all(obj1[k] <= obj2[k] for k in obj1)
        strictly_worse = any(obj1[k] < obj2[k] for k in obj1)
        return worse_equal and strictly_worse
    
    def get_pareto_front(self, 
                        population: List[List[str]],
                        objectives: Dict[str, Dict[str, float]]) -> List[List[str]]:
        """Get Pareto frontier from population."""
        pareto_front = []
        for arch1 in population:
            dominated = False
            for arch2 in population:
                if arch1 != arch2:
                    if self.is_dominated(objectives[str(arch1)], 
                                       objectives[str(arch2)]):
                        dominated = True
                        break
            if not dominated:
                pareto_front.append(arch1)
        return pareto_front
    
    def crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """Perform crossover between two parent architectures."""
        crossover_point = np.random.randint(1, self.num_layers)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def mutate(self, architecture: List[str], mutation_rate: float = 0.1) -> List[str]:
        """Mutate an architecture."""
        mutated = architecture.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = np.random.choice(self.operations)
        return mutated
    
    def search(self, 
              model: nn.Module,
              valid_loader: DataLoader,
              device: torch.device) -> List[List[str]]:
        """Perform multi-objective architecture search."""
        for generation in range(self.num_generations):
            print(f"\nGeneration {generation + 1}/{self.num_generations}")
            
            # Evaluate current population
            for arch in self.population:
                if str(arch) not in self.objectives:
                    self.evaluate_architecture(arch, model, valid_loader, device)
            
            # Get Pareto front
            pareto_front = self.get_pareto_front(self.population, self.objectives)
            
            # Create new population
            new_population = pareto_front.copy()
            
            # Fill rest of population with crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = np.random.choice(pareto_front, 2, replace=True)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            
            # Print statistics
            print(f"Pareto front size: {len(pareto_front)}")
            for arch in pareto_front[:3]:  # Show top 3 architectures
                obj = self.objectives[str(arch)]
                print(f"Architecture: acc={obj['accuracy']:.3f}, "
                      f"lat={-obj['latency']:.1f}ms, "
                      f"flops={-obj['flops']:.1f}M")
        
        return self.get_pareto_front(self.population, self.objectives)
    
    def visualize_pareto_front(self, pareto_front: List[List[str]]):
        """Visualize Pareto front in 2D projections."""
        objectives = [self.objectives[str(arch)] for arch in pareto_front]
        
        # Create 2D projections
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        metrics = ['accuracy', 'latency', 'flops', 'params']
        
        for i, metric1 in enumerate(metrics[:2]):
            for j, metric2 in enumerate(metrics[2:]):
                ax = axes[i, j]
                x = [-obj[metric1] if metric1 != 'accuracy' else obj[metric1] 
                     for obj in objectives]
                y = [-obj[metric2] if metric2 != 'accuracy' else obj[metric2] 
                     for obj in objectives]
                
                ax.scatter(x, y)
                ax.set_xlabel(metric1)
                ax.set_ylabel(metric2)
                ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data (similar to previous recipes)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    valid_data = datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform)
    
    valid_loader = DataLoader(
        valid_data, batch_size=128, shuffle=False, num_workers=2)
    
    # Initialize search
    nas = MultiObjectiveNAS(
        num_layers=8,
        population_size=100,
        num_generations=50
    )
    
    # Create a simple model for evaluation
    model = SimpleCNN().to(device)
    
    # Perform search
    pareto_front = nas.search(model, valid_loader, device)
    
    # Visualize results
    nas.visualize_pareto_front(pareto_front)

class SimpleCNN(nn.Module):
    """Simple CNN for architecture evaluation."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.ops = nn.ModuleDict({
            'conv3x3': nn.Conv2d(16, 16, 3, padding=1),
            'conv5x5': nn.Conv2d(16, 16, 5, padding=2),
            'conv7x7': nn.Conv2d(16, 16, 7, padding=3),
            'maxpool3x3': nn.MaxPool2d(3, stride=1, padding=1),
            'avgpool3x3': nn.AvgPool2d(3, stride=1, padding=1),
            'skip_connect': nn.Identity()
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

if __name__ == '__main__':
    main()
