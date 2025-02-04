import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SharedCNN(nn.Module):
    """Simple CNN with shared parameters."""
    def __init__(self, num_layers=4, in_channels=3, num_classes=10):
        super().__init__()
        self.num_layers = num_layers
        
        # Shared operations pool
        self.ops = nn.ModuleDict({
            'conv3x3': nn.Conv2d(64, 64, 3, padding=1),
            'conv5x5': nn.Conv2d(64, 64, 5, padding=2),
            'maxpool': nn.MaxPool2d(3, stride=1, padding=1)
        })
        self.op_names = list(self.ops.keys())
        
        # Input and output layers
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, architecture):
        x = self.stem(x)
        
        # Apply operations according to architecture
        for op_name in architecture:
            x = self.ops[op_name](x)
            x = F.relu(x)
        
        return self.classifier(x)

class Controller(nn.Module):
    """Controller that samples architectures."""
    def __init__(self, num_layers=4, hidden_size=100):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_ops = 3  # number of possible operations
        
        # LSTM for generating architectures
        self.lstm = nn.LSTMCell(self.num_ops, hidden_size)
        self.classifier = nn.Linear(hidden_size, self.num_ops)
        
        # Map indices to operation names
        self.idx_to_ops = ['conv3x3', 'conv5x5', 'maxpool']

    def forward(self, batch_size=1):
        architectures = []
        log_probs = []
        entropies = []
        
        # Initialize LSTM states
        h_t = torch.zeros(batch_size, self.hidden_size).cuda()
        c_t = torch.zeros(batch_size, self.hidden_size).cuda()
        inputs = torch.zeros(batch_size, self.num_ops).cuda()
        
        # Generate architecture for each layer
        for _ in range(self.num_layers):
            h_t, c_t = self.lstm(inputs, (h_t, c_t))
            logits = self.classifier(h_t)
            probs = F.softmax(logits, dim=-1)
            
            # Sample actions and calculate log probabilities
            actions = torch.multinomial(probs, 1).squeeze(-1)
            log_prob = F.log_softmax(logits, dim=-1)
            selected_log_prob = log_prob[range(batch_size), actions]
            
            # Calculate entropy for exploration
            entropy = -(log_prob * probs).sum(dim=-1)
            
            # Convert indices to operation names
            arch_steps = [self.idx_to_ops[idx] for idx in actions.cpu().numpy()]
            
            architectures.append(arch_steps[0] if batch_size == 1 else arch_steps)
            log_probs.append(selected_log_prob)
            entropies.append(entropy)
        
        return architectures, torch.stack(log_probs), torch.stack(entropies)

def get_data_loaders(batch_size=128):
    """Create CIFAR-10 data loaders for training and validation."""
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                   download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_shared_cnn(model, controller, train_loader, optimizer, device):
    """Train the shared CNN for one epoch."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Sample architecture
        architecture, _, _ = controller(batch_size=1)
        
        optimizer.zero_grad()
        output = model(data, architecture)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Loss: {loss.item():.4f}')

def train_controller(model, controller, valid_loader, optimizer, device):
    """Train the controller using REINFORCE."""
    model.eval()
    controller.train()
    
    # Sample multiple architectures and evaluate them
    batch_size = 5
    architectures, log_probs, entropies = controller(batch_size=batch_size)
    
    # Calculate rewards for each architecture
    rewards = []
    with torch.no_grad():
        for arch_idx in range(batch_size):
            correct = 0
            total = 0
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, [architectures[i][arch_idx] for i in range(len(architectures))])
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            reward = correct / total
            rewards.append(reward)
    
    # Calculate loss and update controller
    rewards = torch.tensor(rewards).to(device)
    baseline = rewards.mean()
    rewards = rewards - baseline
    
    # Policy gradient loss
    loss = -torch.mean(log_probs.sum(dim=0) * rewards)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = get_data_loaders()
    
    # Initialize models
    shared_cnn = SharedCNN().to(device)
    controller = Controller().to(device)
    
    # Optimizers
    shared_optim = optim.SGD(shared_cnn.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=1e-4)
    ctrl_optim = optim.Adam(controller.parameters(), lr=0.00035)
    
    # Training loop
    for epoch in range(50):
        print(f"\nEpoch {epoch+1}")
        
        # Phase 1: Train SharedCNN
        train_shared_cnn(shared_cnn, controller, train_loader, shared_optim, device)
        
        # Phase 2: Train Controller
        ctrl_loss = train_controller(shared_cnn, controller, valid_loader, ctrl_optim, device)
        print(f'Controller Loss: {ctrl_loss:.4f}')

if __name__ == "__main__":
    main()
