import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):
    """Mixture of operations in the search space."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ops = nn.ModuleDict({
            # Basic convolutions
            'conv3x3': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            'conv5x5': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, padding=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            # Depthwise separable convolutions
            'sep_conv3x3': nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            # Dilated convolutions
            'dilated_conv3x3': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            # Pooling operations
            'avg_pool': nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            'max_pool': nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            # Identity for skip connections
            'identity': nn.Identity() if in_channels == out_channels else None
        })

    def forward(self, x, op_name):
        return self.ops[op_name](x)

class Cell(nn.Module):
    """A cell is a basic building block of the network."""
    def __init__(self, num_nodes, channels):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Create mixed operations for each node
        self.ops = nn.ModuleList()
        for i in range(num_nodes):
            # Each node can take input from previous nodes
            node_ops = nn.ModuleList()
            for j in range(i + 1):
                op = MixedOp(channels, channels)
                node_ops.append(op)
            self.ops.append(node_ops)

    def forward(self, x, cell_arch):
        states = [x]
        
        for i in range(self.num_nodes):
            node_inputs = []
            for j, op_name in enumerate(cell_arch[i]):
                if op_name != 'none':  # Skip if operation is none
                    h = self.ops[i][j](states[j], op_name)
                    node_inputs.append(h)
            
            # Combine inputs (sum or concatenate)
            s = sum(node_inputs)
            states.append(s)
        
        return torch.cat(states[1:], dim=1)  # Concatenate all intermediate nodes

class SearchableNetwork(nn.Module):
    """Network composed of searchable cells."""
    def __init__(self, num_cells=8, num_nodes=4, channels=16, num_classes=10):
        super().__init__()
        self.num_cells = num_cells
        
        # Initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # Stack of cells
        self.cells = nn.ModuleList()
        for i in range(num_cells):
            # Double channels at reduction cells
            if i in [num_cells//3, 2*num_cells//3]:
                channels *= 2
            cell = Cell(num_nodes, channels)
            self.cells.append(cell)
        
        # Classification head
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels * num_nodes, num_classes)

    def forward(self, x, architecture):
        x = self.stem(x)
        
        for i, cell in enumerate(self.cells):
            x = cell(x, architecture[i])
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def analyze_search_space(num_cells=8, num_nodes=4):
    """Analyze the size and properties of the search space."""
    ops = ['conv3x3', 'conv5x5', 'sep_conv3x3', 'dilated_conv3x3', 
           'avg_pool', 'max_pool', 'identity', 'none']
    
    # Calculate search space size
    choices_per_node = len(ops)
    connections_per_cell = sum(range(num_nodes))
    total_choices = (choices_per_node ** connections_per_cell) ** num_cells
    
    print(f"Search Space Analysis:")
    print(f"Operations available: {len(ops)}")
    print(f"Connections per cell: {connections_per_cell}")
    print(f"Total architecture choices: {total_choices}")
    print(f"Log10(search space size): {torch.log10(torch.tensor(total_choices)):.2f}")

def main():
    # Create a sample network
    model = SearchableNetwork()
    
    # Generate a random architecture
    num_cells = 8
    num_nodes = 4
    ops = ['conv3x3', 'conv5x5', 'sep_conv3x3', 'dilated_conv3x3', 
           'avg_pool', 'max_pool', 'identity']
    
    architecture = []
    for _ in range(num_cells):
        cell_arch = []
        for i in range(num_nodes):
            node_ops = [ops[torch.randint(len(ops), (1,)).item()] 
                       for _ in range(i + 1)]
            cell_arch.append(node_ops)
        architecture.append(cell_arch)
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x, architecture)
    print(f"Output shape: {output.shape}")
    
    # Analyze search space
    analyze_search_space()

if __name__ == "__main__":
    main()
