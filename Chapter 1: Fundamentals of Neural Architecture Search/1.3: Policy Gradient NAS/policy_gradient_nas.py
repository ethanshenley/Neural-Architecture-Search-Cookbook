import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SharedCNN(nn.Module):
    """Shared CNN that will be used to evaluate architectures."""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Shared operations
        self.ops = nn.ModuleDict({
            'conv3x3': nn.Conv2d(16, 16, 3, padding=1),
            'conv5x5': nn.Conv2d(16, 16, 5, padding=2),
            'maxpool': nn.MaxPool2d(3, stride=1, padding=1),
            'avgpool': nn.AvgPool2d(3, stride=1, padding=1),
            'sep_conv3x3': nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1, groups=16),
                nn.Conv2d(16, 16, 1)
            ),
            'identity': nn.Identity()
        })
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x, architecture):
        x = self.stem(x)
        
        # Apply operations according to architecture
        for op_name in architecture:
            x = self.ops[op_name](x)
            x = F.relu(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class PolicyController(nn.Module):
    """Controller that generates architectures using policy gradients."""
    def __init__(self, 
                 num_layers=4,
                 hidden_size=100,
                 num_operations=6,
                 temperature=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_operations = num_operations
        self.temperature = temperature

        # LSTM controller
        self.lstm = nn.LSTMCell(
            input_size=self.num_operations,
            hidden_size=self.hidden_size
        )
        
        # Prediction layers
        self.operation_decoder = nn.Linear(hidden_size, num_operations)
        
        # Operation embeddings
        self.op_embedding = nn.Embedding(num_operations, num_operations)
        
        # Available operations
        self.operations = [
            'conv3x3', 'conv5x5', 'maxpool', 'avgpool', 
            'sep_conv3x3', 'identity'
        ]

    def forward(self, batch_size=1, device=None):
        """Sample architectures from the policy."""
        if device is None:
            device = next(self.parameters()).device
            
        inputs = torch.zeros(batch_size, self.num_operations).to(device)
        h_t = torch.zeros(batch_size, self.hidden_size).to(device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(device)
        
        actions = []
        log_probs = []
        entropies = []
        
        for layer in range(self.num_layers):
            # LSTM forward pass
            h_t, c_t = self.lstm(inputs, (h_t, c_t))
            
            # Predict operation
            logits = self.operation_decoder(h_t) / self.temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample action and calculate log probability
            if self.training:
                action = torch.multinomial(probs, 1).squeeze(-1)
            else:
                action = probs.argmax(dim=-1)
            
            log_prob = F.log_softmax(logits, dim=-1)
            selected_log_prob = log_prob.gather(1, action.unsqueeze(-1))
            entropy = -(log_prob * probs).sum(dim=-1, keepdim=True)
            
            # Prepare input for next step
            inputs = self.op_embedding(action)
            
            # Store results
            actions.append(action)
            log_probs.append(selected_log_prob)
            entropies.append(entropy)
        
        return (torch.stack(actions), 
                torch.stack(log_probs), 
                torch.stack(entropies))

class RewardPredictor:
    """Predicts rewards for architectures using a moving average baseline."""
    def __init__(self, window_size=20):
        self.rewards = deque(maxlen=window_size)
        
    def get_reward(self, accuracy):
        """Convert accuracy to reward with baseline subtraction."""
        reward = accuracy
        baseline = np.mean(self.rewards) if self.rewards else 0
        self.rewards.append(reward)
        return reward - baseline

def get_data_loaders(batch_size=128):
    """Create CIFAR-10 data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    valid_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, valid_loader

def train_shared_cnn(model, architecture, train_loader, optimizer, device):
    """Train the shared CNN for one epoch."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, architecture)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Loss: {loss.item():.4f}')

def evaluate_architecture(model, architecture, valid_loader, device):
    """Evaluate an architecture on the validation set."""
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
    train_loader, valid_loader = get_data_loaders()
    
    # Initialize models
    controller = PolicyController().to(device)
    shared_cnn = SharedCNN().to(device)
    reward_predictor = RewardPredictor()
    
    # Optimizers
    controller_optimizer = torch.optim.Adam(controller.parameters(), lr=0.0035)
    shared_optimizer = torch.optim.SGD(
        shared_cnn.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    
    # Training loop
    for epoch in range(50):
        print(f"\nEpoch {epoch}")
        
        # Sample architectures and calculate rewards
        controller.train()
        actions, log_probs, entropies = controller(batch_size=5, device=device)
        
        rewards = []
        for arch_idx in range(actions.size(1)):
            # Convert actions to architecture
            architecture = [
                controller.operations[act[arch_idx].item()]
                for act in actions
            ]
            
            # Train shared CNN with this architecture
            train_shared_cnn(
                shared_cnn, architecture, train_loader, shared_optimizer, device)
            
            # Evaluate architecture
            accuracy = evaluate_architecture(
                shared_cnn, architecture, valid_loader, device)
            reward = reward_predictor.get_reward(accuracy)
            rewards.append(reward)
            
            print(f"Architecture {arch_idx + 1} accuracy: {accuracy:.4f}")
        
        # Update controller
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Policy gradient loss
        pg_loss = -(log_probs.sum(dim=0) * rewards).mean()
        entropy_loss = -entropies.mean() * 0.0001
        loss = pg_loss + entropy_loss
        
        controller_optimizer.zero_grad()
        loss.backward()
        controller_optimizer.step()
        
        print(f"Controller loss: {loss.item():.4f}")
        
        # Sample best architecture
        controller.eval()
        with torch.no_grad():
            actions, _, _ = controller(batch_size=1, device=device)
            best_architecture = [
                controller.operations[act[0].item()]
                for act in actions
            ]
            print(f"Best architecture: {best_architecture}")

if __name__ == "__main__":
    main()
