import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from collections import OrderedDict
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

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

class ArchitectureDataset(Dataset):
    """Dataset for architecture-performance pairs."""
    def __init__(self, architectures: List[List[str]], performances: List[float]):
        self.architectures = architectures
        self.performances = performances
        self.encoder = ArchitectureEncoder()
        
        # Normalize performance values
        self.scaler = StandardScaler()
        self.performances = self.scaler.fit_transform(
            np.array(performances).reshape(-1, 1)).squeeze()
    
    def __len__(self):
        return len(self.architectures)
    
    def __getitem__(self, idx):
        arch = self.architectures[idx]
        encoding = self.encoder.encode(arch)
        performance = self.performances[idx]
        return encoding, performance

class ArchitectureEncoder(nn.Module):
    """Encodes architecture into continuous representation."""
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.op_embedding = nn.Embedding(len(OPERATIONS), hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Available operations
        self.op_to_idx = {op: idx for idx, op in enumerate(OPERATIONS)}
    
    def encode(self, architecture: List[str]) -> torch.Tensor:
        """Convert architecture to tensor representation."""
        # Convert operations to indices
        indices = torch.tensor([self.op_to_idx[op] for op in architecture])
        # Get embeddings
        embeddings = self.op_embedding(indices)
        # Process with LSTM
        output, (hidden, _) = self.lstm(embeddings.unsqueeze(0))
        return hidden.squeeze()

class EnsemblePredictor(nn.Module):
    """Ensemble of MLPs for performance prediction with uncertainty."""
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_ensemble: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.num_ensemble = num_ensemble
        
        # Create ensemble of MLPs
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            ) for _ in range(num_ensemble)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and variance predictions."""
        predictions = torch.cat([model(x) for model in self.ensemble], dim=1)
        mean = predictions.mean(dim=1, keepdim=True)
        var = predictions.var(dim=1, keepdim=True)
        return mean, var

class NeuralPredictor:
    """Neural network-based architecture performance predictor."""
    def __init__(self,
                 input_size: int = 64,
                 hidden_size: int = 128,
                 num_ensemble: int = 5,
                 learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = ArchitectureEncoder(hidden_size=input_size).to(self.device)
        self.predictor = EnsemblePredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_ensemble=num_ensemble
        ).to(self.device)
        
        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=learning_rate)
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=learning_rate)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, 
                  train_loader: DataLoader,
                  epoch: int):
        """Single training epoch."""
        self.encoder.train()
        self.predictor.train()
        total_loss = 0
        
        for batch_idx, (encodings, targets) in enumerate(train_loader):
            encodings, targets = encodings.to(self.device), targets.to(self.device)
            
            # Forward pass
            encoded = self.encoder(encodings)
            mean_pred, var_pred = self.predictor(encoded)
            
            # Negative log likelihood loss
            loss = self.gaussian_nll_loss(targets.view(-1, 1), mean_pred, var_pred)
            
            # Backward pass
            self.encoder_optimizer.zero_grad()
            self.predictor_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            self.predictor_optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\t'
                      f'Loss: {loss.item():.6f}')
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, valid_loader: DataLoader) -> float:
        """Validate predictor."""
        self.encoder.eval()
        self.predictor.eval()
        total_loss = 0
        
        for encodings, targets in valid_loader:
            encodings, targets = encodings.to(self.device), targets.to(self.device)
            encoded = self.encoder(encodings)
            mean_pred, var_pred = self.predictor(encoded)
            loss = self.gaussian_nll_loss(targets.view(-1, 1), mean_pred, var_pred)
            total_loss += loss.item()
        
        return total_loss / len(valid_loader)
    
    def gaussian_nll_loss(self,
                         targets: torch.Tensor,
                         mean: torch.Tensor,
                         var: torch.Tensor) -> torch.Tensor:
        """Gaussian negative log likelihood loss."""
        return (torch.log(var) + (targets - mean)**2 / var).mean()
    
    def predict(self, architecture: List[str]) -> Tuple[float, float]:
        """Predict performance and uncertainty for architecture."""
        self.encoder.eval()
        self.predictor.eval()
        
        with torch.no_grad():
            encoding = self.encoder.encode(architecture).to(self.device)
            mean, var = self.predictor(encoding.unsqueeze(0))
            return mean.item(), var.item()
    
    def acquisition_function(self,
                           mean: float,
                           var: float,
                           best_value: float,
                           kappa: float = 2.0) -> float:
        """Upper Confidence Bound acquisition function."""
        return mean + kappa * np.sqrt(var)

class NeuralPredictorSearch:
    """Architecture search using neural predictor."""
    def __init__(self,
                 predictor: NeuralPredictor,
                 eval_model: SimpleCNN,
                 num_iterations: int = 100,
                 population_size: int = 50):
        self.predictor = predictor
        self.eval_model = eval_model
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.best_architecture = None
        self.best_performance = float('-inf')
    
    def generate_random_architecture(self, length: int = 8) -> List[str]:
        """Generate random architecture."""
        return [random.choice(OPERATIONS) for _ in range(length)]
    
    def search(self, valid_loader: DataLoader) -> Tuple[List[str], float]:
        """Perform architecture search."""
        for iteration in range(self.num_iterations):
            print(f"\nIteration {iteration + 1}/{self.num_iterations}")
            
            # Generate candidate architectures
            candidates = [
                self.generate_random_architecture()
                for _ in range(self.population_size)
            ]
            
            # Predict performance for all candidates
            predictions = []
            for arch in candidates:
                mean, var = self.predictor.predict(arch)
                acq_value = self.predictor.acquisition_function(
                    mean, var, self.best_performance)
                predictions.append((arch, acq_value))
            
            # Select best candidate
            best_candidate = max(predictions, key=lambda x: x[1])[0]
            
            # Evaluate best candidate
            performance = evaluate_architecture(
                architecture=best_candidate,
                model=self.eval_model,
                valid_loader=valid_loader
            )
            
            if performance > self.best_performance:
                self.best_architecture = best_candidate
                self.best_performance = performance
                print(f"New best architecture found!")
                print(f"Architecture: {self.best_architecture}")
                print(f"Performance: {self.best_performance:.4f}")
        
        return self.best_architecture, self.best_performance

def plot_learning_curves(predictor: NeuralPredictor):
    """Plot training and validation learning curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(predictor.train_losses, label='Train Loss')
    plt.plot(predictor.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neural Predictor Learning Curves')
    plt.legend()
    plt.show()

# Constants
OPERATIONS = [
    'conv3x3', 'conv5x5', 'maxpool3x3',
    'avgpool3x3', 'skip', 'none'
]

def evaluate_architecture(
    architecture: List[str],
    model: SimpleCNN,
    valid_loader: DataLoader
) -> float:
    """Evaluate an architecture on the validation set."""
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
    
    # Create evaluation model
    eval_model = SimpleCNN().to(device)
    
    # Create synthetic training data
    num_samples = 1000
    architectures = [
        [random.choice(OPERATIONS) for _ in range(8)]
        for _ in range(num_samples)
    ]
    
    # Create datasets and loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    val_data = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    
    # Evaluate architectures to get real performances
    performances = []
    for arch in architectures:
        perf = evaluate_architecture(
            architecture=arch,
            model=eval_model,
            valid_loader=val_loader
        )
        performances.append(perf)
    
    # Create predictor dataset
    dataset = ArchitectureDataset(architectures, performances)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    predictor_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    predictor_val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create and train predictor
    predictor = NeuralPredictor()
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = predictor.train_step(predictor_train_loader, epoch)
        val_loss = predictor.validate(predictor_val_loader)
        
        predictor.train_losses.append(train_loss)
        predictor.val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
    
    # Plot learning curves
    plot_learning_curves(predictor)
    
    # Perform architecture search
    searcher = NeuralPredictorSearch(
        predictor=predictor,
        eval_model=eval_model
    )
    best_arch, best_perf = searcher.search(val_loader)
    
    print("\nSearch completed!")
    print(f"Best architecture: {best_arch}")
    print(f"Predicted performance: {best_perf:.4f}")

if __name__ == "__main__":
    main()
