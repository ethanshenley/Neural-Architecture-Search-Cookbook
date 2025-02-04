import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import namedtuple
from torchvision import datasets, transforms
import argparse

# Define the set of primitive operations
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

class ReLUConvBN(nn.Module):
    """Basic Conv-BN-ReLU operations."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, 
                     padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=True)
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    """Dilated separable convolution."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, 
                     padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=True)
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    """Separable convolution."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_out, C_out, kernel_size, stride=1, 
                     padding=padding, groups=C_out, bias=False),
            nn.Conv2d(C_out, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=True)
        )

    def forward(self, x):
        return self.op(x)

class FactorizedReduce(nn.Module):
    """Reduce feature map size by factorized pointwise convolution."""
    def __init__(self, C_in, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=True)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

# Operation dictionary
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2)
}

class Identity(nn.Module):
    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)

def get_args():
    parser = argparse.ArgumentParser('DARTS')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.025)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--init_channels', type=int, default=16)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4)
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3)
    parser.add_argument('--report_freq', type=int, default=50)
    return parser.parse_args()

def get_data_loaders(args):
    """Create CIFAR-10 data loaders for training and validation."""
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(0.5 * num_train))
    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)
    
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=2)
    
    return train_queue, valid_queue

class Genotype:
    """Helper class for storing gene information."""
    def __init__(self, normal=None, normal_concat=None, reduce=None, reduce_concat=None):
        self.normal = normal
        self.normal_concat = normal_concat
        self.reduce = reduce
        self.reduce_concat = reduce_concat

    def __str__(self):
        return f'Genotype(normal={self.normal}, normal_concat={self.normal_concat}, '\
               f'reduce={self.reduce}, reduce_concat={self.reduce_concat})'

class MixedOp(nn.Module):
    """Continuously-relaxed mixed operation."""
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """Forward computation using weighted sum of operations."""
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    """Cell structure containing mixed operations."""
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        self.steps = steps  # Number of intermediate nodes

        # Input preprocessing
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        # DAG connections
        self._ops = nn.ModuleList()
        self._compile(C, reduction)

    def _compile(self, C, reduction):
        """Create the DAG connections."""
        self.edge_keys = []
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)
                self.edge_keys.append((j, i + 2))

    def forward(self, s0, s1, weights):
        """Forward computation through the cell."""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        
        # Process intermediate nodes
        for i in range(self.steps):
            s = sum(self._ops[offset + j](h, weights[offset + j])
                   for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self.multiplier:], dim=1)

class Network(nn.Module):
    """Main network composed of cells."""
    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        for i in range(layers):
            reduction = i in [layers // 3, 2 * layers // 3]
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        # Initialize architecture parameters
        k = sum(2 + i for i in range(steps))
        num_ops = len(PRIMITIVES)
        self.alphas = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)

    def forward(self, input):
        """Forward computation."""
        s0 = s1 = self.stem(input)
        for cell in self.cells:
            weights = F.softmax(self.alphas, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def arch_parameters(self):
        return [self.alphas]

class Architect:
    """Handles architecture optimization."""
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        """Performs one step of architecture optimization."""
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        """Compute architecture gradients."""
        loss = self.model._criterion(self.model(input_valid), target_valid)
        loss.backward()

def train_model(model, architect, train_queue, valid_queue, optimizer, args):
    """Main training loop."""
    for epoch in range(args.epochs):
        # Train
        model.train()
        for step, (input_train, target_train) in enumerate(train_queue):
            input_train = input_train.cuda()
            target_train = target_train.cuda(non_blocking=True)
            
            # Get validation data for architecture update
            input_valid, target_valid = next(iter(valid_queue))
            input_valid = input_valid.cuda()
            target_valid = target_valid.cuda(non_blocking=True)
            
            # Phase 1: Architecture optimization
            architect.step(input_train, target_train, input_valid, target_valid, 
                         args.learning_rate, optimizer)
            
            # Phase 2: Network weight optimization
            optimizer.zero_grad()
            logits = model(input_train)
            loss = model._criterion(logits, target_train)
            loss.backward()
            optimizer.step()
            
            if step % args.report_freq == 0:
                print(f'train step {step} loss {loss.item()}')

def main():
    # Setup
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    criterion = nn.CrossEntropyLoss()
    model = Network(args.init_channels, args.num_classes, args.layers, criterion).to(device)
    architect = Architect(model, args)
    
    # Setup optimizers
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs)
    )
    
    # Get data loaders
    train_queue, valid_queue = get_data_loaders(args)
    
    # Training loop
    for epoch in range(args.epochs):
        scheduler.step()
        
        # Training
        train_model(model, architect, train_queue, valid_queue, optimizer, args)
        
        # Print current architecture
        genotype = model.genotype()
        print(f'\nEpoch {epoch}: Current architecture:\n{genotype}')

if __name__ == '__main__':
    main()
