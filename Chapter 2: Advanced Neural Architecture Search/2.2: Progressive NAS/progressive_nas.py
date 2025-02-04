import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import Ridge
import copy

class SurrogatePredictor:
    """Predicts architecture performance without full training."""
    def __init__(self, encoding_size: int):
        self.predictor = Ridge(alpha=1.0)
        self.encoding_size = encoding_size
        self.trained = False
        
    def encode_architecture(self, architecture: List[str]) -> np.ndarray:
        """Encode architecture into feature vector."""
        # Simple one-hot encoding for operations
        ops = ['conv3x3', 'conv5x5', 'maxpool3x3', 'avgpool3x3', 'skip']
        encoding = []
        for op in architecture:
            one_hot = [1.0 if op == available_op else 0.0 for available_op in ops]
            encoding.extend(one_hot)
        return np.array(encoding)
    
    def fit(self, architectures: List[List[str]], performances: List[float]):
        """Train predictor on architecture-performance pairs."""
        X = np.array([self.encode_architecture(arch) for arch in architectures])
        y = np.array(performances)
        self.predictor.fit(X, y)
        self.trained = True
    
    def predict(self, architecture: List[str]) -> float:
        """Predict performance of an architecture."""
        if not self.trained:
            return 0.0
        x = self.encode_architecture(architecture).reshape(1, -1)
        return self.predictor.predict(x)[0]

class ProgressiveCell(nn.Module):
    """Cell that progressively adds operations."""
    def __init__(self, C_in: int, C_out: int, stride: int):
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
    
    def forward(self, x: torch.Tensor, op_name: str) -> torch.Tensor:
        if op_name == 'skip' and self.ops['skip'] is None:
            return torch.zeros_like(x)
        return self.ops[op_name](x)

class ProgressiveNetwork(nn.Module):
    """Network that grows progressively during search."""
    def __init__(self, 
                 num_classes: int = 10,
                 init_channels: int = 16,
                 num_cells: int = 8):
        super().__init__()
        self.num_cells = num_cells
        C = init_channels
        
        # Initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU()
        )
        
        # Progressive cells
        self.cells = nn.ModuleList()
        for i in range(num_cells):
            if i in [num_cells//3, 2*num_cells//3]:
                C *= 2
                stride = 2
            else:
                stride = 1
            cell = ProgressiveCell(C, C, stride)
            self.cells.append(cell)
        
        # Classification head
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, num_classes)
    
    def forward(self, x: torch.Tensor, architecture: List[str]) -> torch.Tensor:
        x = self.stem(x)
        
        for cell, op_name in zip(self.cells, architecture):
            x = cell(x, op_name)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ProgressiveNAS:
    """Progressive Neural Architecture Search."""
    def __init__(self,
                 model: ProgressiveNetwork,
                 num_classes: int = 10,
                 beam_size: int = 8,
                 max_complexity: int = 4):
        self.model = model
        self.num_classes = num_classes
        self.beam_size = beam_size
        self.max_complexity = max_complexity
        
        # Available operations at each complexity level
        self.ops_by_complexity = [
            ['conv3x3'],  # Level 1
            ['conv3x3', 'maxpool3x3'],  # Level 2
            ['conv3x3', 'maxpool3x3', 'skip'],  # Level 3
            ['conv3x3', 'conv5x5', 'maxpool3x3', 'avgpool3x3', 'skip']  # Level 4
        ]
        
        # Initialize surrogate predictor
        self.predictor = SurrogatePredictor(encoding_size=5*model.num_cells)
        
        # Track best architectures
        self.history = {i: [] for i in range(1, max_complexity + 1)}
    
    def train_predictor(self, 
                       architectures: List[List[str]], 
                       valid_loader: DataLoader,
                       device: torch.device):
        """Train surrogate predictor on evaluated architectures."""
        performances = []
        for arch in architectures:
            acc = self.evaluate_architecture(arch, valid_loader, device)
            performances.append(acc)
        
        self.predictor.fit(architectures, performances)
    
    def evaluate_architecture(self,
                            architecture: List[str],
                            valid_loader: DataLoader,
                            device: torch.device) -> float:
        """Evaluate architecture on validation set."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data, architecture)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def search(self, 
              train_loader: DataLoader,
              valid_loader: DataLoader,
              device: torch.device):
        """Perform progressive architecture search."""
        current_archs = [['conv3x3'] * self.model.num_cells]  # Start with simplest
        
        for complexity in range(1, self.max_complexity + 1):
            print(f"\nSearching at complexity level {complexity}")
            candidates = []
            
            # Generate candidates by modifying current architectures
            for base_arch in current_archs:
                for i in range(len(base_arch)):
                    for op in self.ops_by_complexity[complexity-1]:
                        new_arch = copy.deepcopy(base_arch)
                        new_arch[i] = op
                        candidates.append(new_arch)
            
            # Remove duplicates
            candidates = [list(x) for x in set(tuple(x) for x in candidates)]
            
            # Train predictor if we have history
            if complexity > 1:
                prev_archs = []
                for level in range(1, complexity):
                    prev_archs.extend(self.history[level])
                self.train_predictor(prev_archs, valid_loader, device)
            
            # Predict performance for all candidates
            predictions = []
            for arch in candidates:
                if self.predictor.trained:
                    pred = self.predictor.predict(arch)
                else:
                    # For first level, evaluate all candidates
                    pred = self.evaluate_architecture(arch, valid_loader, device)
                predictions.append(pred)
            
            # Select top architectures
            sorted_indices = np.argsort(predictions)[-self.beam_size:]
            current_archs = [candidates[i] for i in sorted_indices]
            
            # Evaluate and store top architectures
            for arch in current_archs:
                acc = self.evaluate_architecture(arch, valid_loader, device)
                self.history[complexity].append((arch, acc))
                print(f"Architecture: {arch}")
                print(f"Accuracy: {acc:.4f}")
            
            # Train model with best architecture
            best_arch = max(self.history[complexity], key=lambda x: x[1])[0]
            self.train_model(best_arch, train_loader, valid_loader, device)
    
    def train_model(self,
                   architecture: List[str],
                   train_loader: DataLoader,
                   valid_loader: DataLoader,
                   device: torch.device,
                   epochs: int = 5):
        """Train model with given architecture."""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.025,
            momentum=0.9,
            weight_decay=3e-4
        )
        
        for epoch in range(epochs):
            self.model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data, architecture)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

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
    
    train_loader = DataLoader(train_data, batch_size=96, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size=96, shuffle=False, num_workers=2)
    
    # Create model and searcher
    model = ProgressiveNetwork().to(device)
    searcher = ProgressiveNAS(model)
    
    # Perform search
    searcher.search(train_loader, valid_loader, device)
    
    # Print final results
    print("\nSearch completed! Best architectures by complexity level:")
    for level in range(1, searcher.max_complexity + 1):
        best_arch, best_acc = max(searcher.history[level], key=lambda x: x[1])
        print(f"\nComplexity level {level}:")
        print(f"Architecture: {best_arch}")
        print(f"Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
