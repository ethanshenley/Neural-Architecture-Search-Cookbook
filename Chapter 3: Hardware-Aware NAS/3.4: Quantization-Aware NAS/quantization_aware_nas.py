import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

class QuantizedConv2d(nn.Conv2d):
    """Quantization-aware convolutional layer."""
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 weight_bits: int = 8,
                 activation_bits: int = 8,
                 **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
        # Quantization parameters
        self.weight_scale = None
        self.activation_scale = None
        self.weight_zero_point = None
        self.activation_zero_point = None
    
    def update_quantization_params(self, weight_scale=None, activation_scale=None):
        """Update quantization parameters."""
        if weight_scale is not None:
            self.weight_scale = weight_scale
        if activation_scale is not None:
            self.activation_scale = activation_scale
    
    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize weights to specified bit-width."""
        if self.weight_scale is None:
            self.weight_scale = weight.abs().max() / (2 ** (self.weight_bits - 1) - 1)
        
        weight_q = torch.round(weight / self.weight_scale)
        weight_q = torch.clamp(weight_q, 
                             -2 ** (self.weight_bits - 1),
                             2 ** (self.weight_bits - 1) - 1)
        return weight_q * self.weight_scale
    
    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activations to specified bit-width."""
        if self.activation_scale is None:
            self.activation_scale = x.abs().max() / (2 ** (self.activation_bits - 1) - 1)
        
        x_q = torch.round(x / self.activation_scale)
        x_q = torch.clamp(x_q,
                         -2 ** (self.activation_bits - 1),
                         2 ** (self.activation_bits - 1) - 1)
        return x_q * self.activation_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization."""
        # Quantize input
        x_q = self.quantize_activation(x)
        
        # Quantize weights
        w_q = self.quantize_weight(self.weight)
        
        # Compute convolution with quantized values
        return F.conv2d(x_q, w_q, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

class QuantizationPredictor:
    """Predicts accuracy impact of quantization."""
    def __init__(self):
        # Impact factors for different bit-widths (based on empirical studies)
        self.accuracy_impact = {
            8: 0.99,  # ~1% accuracy drop
            6: 0.97,  # ~3% accuracy drop
            4: 0.92,  # ~8% accuracy drop
            2: 0.80   # ~20% accuracy drop
        }
        
        # Operation sensitivity to quantization
        self.op_sensitivity = {
            'conv3x3': 1.0,
            'conv5x5': 1.2,
            'conv7x7': 1.3,
            'maxpool3x3': 0.7,
            'avgpool3x3': 0.8,
            'skip': 0.5
        }
    
    def predict_accuracy_drop(self,
                            architecture: List[str],
                            bit_widths: List[int]) -> float:
        """Predict accuracy drop from quantization."""
        total_impact = 1.0
        
        for op, bits in zip(architecture, bit_widths):
            impact = self.accuracy_impact[bits]
            sensitivity = self.op_sensitivity[op]
            total_impact *= (impact ** sensitivity)
        
        return 1.0 - total_impact

class QuantizationAwareNAS:
    """Neural Architecture Search with quantization awareness."""
    def __init__(self,
                 accuracy_threshold: float,
                 min_bit_width: int = 4,
                 max_bit_width: int = 8):
        self.accuracy_threshold = accuracy_threshold
        self.min_bit_width = min_bit_width
        self.max_bit_width = max_bit_width
        self.quant_predictor = QuantizationPredictor()
        
        self.operations = [
            'conv3x3', 'conv5x5', 'conv7x7',
            'maxpool3x3', 'avgpool3x3', 'skip'
        ]
        
        self.bit_width_choices = [4, 6, 8]
    
    def generate_random_architecture(self,
                                   num_layers: int = 8) -> Tuple[List[str], List[int]]:
        """Generate random architecture with bit-width configurations."""
        architecture = []
        bit_widths = []
        
        for _ in range(num_layers):
            # Sample operation
            op = np.random.choice(self.operations)
            architecture.append(op)
            
            # Sample bit-width
            bits = np.random.choice(self.bit_width_choices)
            bit_widths.append(bits)
        
        return architecture, bit_widths
    
    def mutate_architecture(self,
                          architecture: List[str],
                          bit_widths: List[int]) -> Tuple[List[str], List[int]]:
        """Mutate architecture and bit-widths."""
        new_architecture = architecture.copy()
        new_bit_widths = bit_widths.copy()
        
        # Randomly choose mutation type
        mutation_type = np.random.choice(['operation', 'bit_width'])
        
        if mutation_type == 'operation':
            # Mutate random operation
            idx = np.random.randint(len(architecture))
            new_architecture[idx] = np.random.choice(self.operations)
        else:
            # Mutate random bit-width
            idx = np.random.randint(len(bit_widths))
            current_bits = bit_widths[idx]
            choices = [b for b in self.bit_width_choices if b != current_bits]
            new_bit_widths[idx] = np.random.choice(choices)
        
        return new_architecture, new_bit_widths

class QuantizedNet(nn.Module):
    """Neural network with quantization-aware design."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            QuantizedConv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Dynamic operations with quantization
        self.ops = {
            'conv3x3': lambda c_in, c_out, bits: QuantizedConv2d(
                c_in, c_out, 3, padding=1, weight_bits=bits, activation_bits=bits),
            'conv5x5': lambda c_in, c_out, bits: QuantizedConv2d(
                c_in, c_out, 5, padding=2, weight_bits=bits, activation_bits=bits),
            'conv7x7': lambda c_in, c_out, bits: QuantizedConv2d(
                c_in, c_out, 7, padding=3, weight_bits=bits, activation_bits=bits),
            'maxpool3x3': lambda c_in, c_out, bits: nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                QuantizedConv2d(c_in, c_out, 1, weight_bits=bits, activation_bits=bits)
                if c_in != c_out else nn.Identity()
            ),
            'avgpool3x3': lambda c_in, c_out, bits: nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                QuantizedConv2d(c_in, c_out, 1, weight_bits=bits, activation_bits=bits)
                if c_in != c_out else nn.Identity()
            ),
            'skip': lambda c_in, c_out, bits: QuantizedConv2d(
                c_in, c_out, 1, weight_bits=bits, activation_bits=bits)
                if c_in != c_out else nn.Identity()
        }
    
    def create_layer(self,
                    op_name: str,
                    c_in: int,
                    c_out: int,
                    bits: int) -> nn.Module:
        """Create quantized layer."""
        return nn.Sequential(
            self.ops[op_name](c_in, c_out, bits),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )
    
    def forward(self,
               x: torch.Tensor,
               architecture: List[str],
               channels: List[int],
               bit_widths: List[int]) -> torch.Tensor:
        x = self.stem(x)
        
        # Apply operations with specified channels and bit-widths
        for i, (op_name, bits) in enumerate(zip(architecture, bit_widths)):
            layer = self.create_layer(op_name, channels[i], channels[i + 1], bits)
            x = layer(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return nn.Linear(channels[-1], 10).to(x.device)(x)

def search_quantized_architecture(
    model: QuantizedNet,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    accuracy_threshold: float,
    device: torch.device,
    num_iterations: int = 1000
) -> Tuple[List[str], List[int], List[int], float]:
    """Perform quantization-aware architecture search."""
    nas = QuantizationAwareNAS(accuracy_threshold=accuracy_threshold)
    
    best_architecture = None
    best_bit_widths = None
    best_channels = None
    best_accuracy = 0
    
    for i in range(num_iterations):
        if i % 100 == 0:
            print(f"Search iteration {i}/{num_iterations}")
        
        # Generate or mutate architecture
        if best_architecture is None:
            architecture, bit_widths = nas.generate_random_architecture()
            channels = [16] * (len(architecture) + 1)  # Simple channel config
        else:
            architecture, bit_widths = nas.mutate_architecture(
                best_architecture, best_bit_widths)
            channels = best_channels
        
        # Predict quantization impact
        accuracy_drop = nas.quant_predictor.predict_accuracy_drop(
            architecture, bit_widths)
        
        # Only evaluate if predicted accuracy is above threshold
        if (1.0 - accuracy_drop) >= accuracy_threshold:
            accuracy = evaluate_architecture(
                model, architecture, channels, bit_widths,
                valid_loader, device)
            
            if accuracy > best_accuracy:
                best_architecture = architecture
                best_bit_widths = bit_widths
                best_channels = channels
                best_accuracy = accuracy
                
                print(f"\nNew best architecture found!")
                print(f"Accuracy: {best_accuracy:.4f}")
                print(f"Average bit-width: {np.mean(best_bit_widths):.1f}")
    
    return best_architecture, best_channels, best_bit_widths, best_accuracy

def evaluate_architecture(
    model: QuantizedNet,
    architecture: List[str],
    channels: List[int],
    bit_widths: List[int],
    valid_loader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate quantized architecture accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, architecture, channels, bit_widths)
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
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=128)
    
    # Create model
    model = QuantizedNet().to(device)
    
    # Set accuracy threshold
    accuracy_threshold = 0.90  # 90% minimum accuracy
    
    # Perform search
    print("Starting quantization-aware architecture search...")
    best_arch, best_channels, best_bits, best_acc = search_quantized_architecture(
        model, train_loader, valid_loader, accuracy_threshold, device)
    
    print("\nSearch completed!")
    print(f"Best architecture: {best_arch}")
    print(f"Bit-width configuration: {best_bits}")
    print(f"Channel configuration: {best_channels}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Average bit-width: {np.mean(best_bits):.1f}")

if __name__ == "__main__":
    main()
