import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

class TransformerSearchSpace:
    """Search space for transformer-specific architectures."""
    def __init__(self):
        # Attention mechanisms
        self.attention_types = [
            'vanilla', 'linear', 'sparse',
            'local', 'longformer', 'performer'
        ]
        
        # Position encoding types
        self.position_encodings = [
            'sinusoidal', 'learned', 'relative',
            'rotary', 'alibi', 'none'
        ]
        
        # Feed-forward network types
        self.ffn_types = [
            'vanilla', 'gated', 'swish',
            'geglu', 'glu', 'dense'
        ]
        
        # Layer normalization positions
        self.norm_positions = [
            'pre', 'post', 'sandwich'
        ]

class TransformerArchitecture:
    """Represents a transformer architecture configuration."""
    def __init__(self,
                 num_layers: int,
                 hidden_size: int,
                 num_heads: List[int],
                 attention_types: List[str],
                 ffn_types: List[str],
                 ffn_ratios: List[float],
                 position_encoding: str,
                 norm_position: str,
                 dropout: float = 0.1):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_types = attention_types
        self.ffn_types = ffn_types
        self.ffn_ratios = ffn_ratios
        self.position_encoding = position_encoding
        self.norm_position = norm_position
        self.dropout = dropout
    
    def compute_flops(self, sequence_length: int) -> int:
        """Estimate computational complexity."""
        total_flops = 0
        
        # Position encoding FLOPs
        if self.position_encoding != 'none':
            total_flops += sequence_length * self.hidden_size
        
        # Per layer FLOPs
        for i in range(self.num_layers):
            # Attention FLOPs
            if self.attention_types[i] == 'vanilla':
                total_flops += 2 * sequence_length**2 * self.hidden_size
            elif self.attention_types[i] == 'linear':
                total_flops += 2 * sequence_length * self.hidden_size
            
            # FFN FLOPs
            ffn_size = int(self.hidden_size * self.ffn_ratios[i])
            total_flops += 2 * sequence_length * self.hidden_size * ffn_size
        
        return total_flops

class TransformerNAS:
    """Neural Architecture Search for transformers."""
    def __init__(self,
                 search_space: TransformerSearchSpace,
                 min_layers: int = 2,
                 max_layers: int = 12,
                 min_heads: int = 1,
                 max_heads: int = 16,
                 min_hidden: int = 128,
                 max_hidden: int = 1024):
        self.search_space = search_space
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.min_heads = min_heads
        self.max_heads = max_heads
        self.min_hidden = min_hidden
        self.max_hidden = max_hidden
    
    def generate_random_architecture(self) -> TransformerArchitecture:
        """Generate random transformer architecture."""
        num_layers = np.random.randint(self.min_layers, self.max_layers + 1)
        hidden_size = np.random.randint(self.min_hidden, self.max_hidden + 1)
        hidden_size = int(np.ceil(hidden_size / 64) * 64)  # Make divisible by 64
        
        # Generate number of heads for each layer
        num_heads = [np.random.randint(self.min_heads, self.max_heads + 1) 
                    for _ in range(num_layers)]
        
        # Ensure hidden size is divisible by number of heads
        num_heads = [min(h, hidden_size // 64) * 64 for h in num_heads]
        
        # Generate layer configurations
        attention_types = [np.random.choice(self.search_space.attention_types) 
                         for _ in range(num_layers)]
        ffn_types = [np.random.choice(self.search_space.ffn_types) 
                    for _ in range(num_layers)]
        ffn_ratios = [np.random.uniform(2.0, 4.0) 
                     for _ in range(num_layers)]
        
        # Global configurations
        position_encoding = np.random.choice(self.search_space.position_encodings)
        norm_position = np.random.choice(self.search_space.norm_positions)
        
        return TransformerArchitecture(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_types=attention_types,
            ffn_types=ffn_types,
            ffn_ratios=ffn_ratios,
            position_encoding=position_encoding,
            norm_position=norm_position
        )
    
    def mutate_architecture(self, 
                          arch: TransformerArchitecture) -> TransformerArchitecture:
        """Mutate existing architecture."""
        new_arch = TransformerArchitecture(
            num_layers=arch.num_layers,
            hidden_size=arch.hidden_size,
            num_heads=arch.num_heads.copy(),
            attention_types=arch.attention_types.copy(),
            ffn_types=arch.ffn_types.copy(),
            ffn_ratios=arch.ffn_ratios.copy(),
            position_encoding=arch.position_encoding,
            norm_position=arch.norm_position,
            dropout=arch.dropout
        )
        
        # Randomly choose mutation type
        mutation_type = np.random.choice([
            'hidden_size', 'num_heads', 'attention',
            'ffn', 'ffn_ratio', 'position', 'norm'
        ])
        
        if mutation_type == 'hidden_size':
            new_size = new_arch.hidden_size * (2 if np.random.random() > 0.5 else 0.5)
            new_size = int(np.clip(new_size, self.min_hidden, self.max_hidden))
            new_size = int(np.ceil(new_size / 64) * 64)  # Make divisible by 64
            new_arch.hidden_size = new_size
        
        elif mutation_type == 'num_heads':
            layer_idx = np.random.randint(new_arch.num_layers)
            new_heads = new_arch.num_heads[layer_idx] * (2 if np.random.random() > 0.5 else 0.5)
            new_heads = int(np.clip(new_heads, self.min_heads, self.max_heads))
            new_heads = min(new_heads, new_arch.hidden_size // 64) * 64
            new_arch.num_heads[layer_idx] = new_heads
        
        elif mutation_type == 'attention':
            layer_idx = np.random.randint(new_arch.num_layers)
            new_arch.attention_types[layer_idx] = np.random.choice(
                self.search_space.attention_types)
        
        elif mutation_type == 'ffn':
            layer_idx = np.random.randint(new_arch.num_layers)
            new_arch.ffn_types[layer_idx] = np.random.choice(
                self.search_space.ffn_types)
        
        elif mutation_type == 'ffn_ratio':
            layer_idx = np.random.randint(new_arch.num_layers)
            new_arch.ffn_ratios[layer_idx] = np.random.uniform(2.0, 4.0)
        
        elif mutation_type == 'position':
            new_arch.position_encoding = np.random.choice(
                self.search_space.position_encodings)
        
        else:  # norm
            new_arch.norm_position = np.random.choice(
                self.search_space.norm_positions)
        
        return new_arch

class TransformerNet(nn.Module):
    """Transformer network with searchable architecture."""
    def __init__(self,
                 vocab_size: int,
                 max_seq_length: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Attention implementations
        self.attention_ops = {
            'vanilla': self._make_vanilla_attention,
            'linear': self._make_linear_attention,
            'sparse': self._make_sparse_attention,
            'local': self._make_local_attention,
            'longformer': self._make_longformer_attention,
            'performer': self._make_performer_attention
        }
        
        # FFN implementations
        self.ffn_ops = {
            'vanilla': self._make_vanilla_ffn,
            'gated': self._make_gated_ffn,
            'swish': self._make_swish_ffn,
            'geglu': self._make_geglu_ffn,
            'glu': self._make_glu_ffn,
            'dense': self._make_dense_ffn
        }
        
        # Position encoding implementations
        self.position_encodings = {
            'sinusoidal': self._make_sinusoidal_encoding,
            'learned': self._make_learned_encoding,
            'relative': self._make_relative_encoding,
            'rotary': self._make_rotary_encoding,
            'alibi': self._make_alibi_encoding,
            'none': lambda x: x
        }
    
    def _make_vanilla_attention(self,
                              hidden_size: int,
                              num_heads: int) -> nn.Module:
        """Create vanilla multi-head attention."""
        return nn.MultiheadAttention(hidden_size, num_heads)
    
    def _make_linear_attention(self,
                             hidden_size: int,
                             num_heads: int) -> nn.Module:
        """Create linear attention (simplified implementation)."""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def _make_sparse_attention(self,
                             hidden_size: int,
                             num_heads: int) -> nn.Module:
        """Create sparse attention (simplified implementation)."""
        return self._make_vanilla_attention(hidden_size, num_heads)
    
    def _make_local_attention(self,
                            hidden_size: int,
                            num_heads: int,
                            window_size: int = 256) -> nn.Module:
        """Create local attention with fixed window."""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def _make_longformer_attention(self,
                                 hidden_size: int,
                                 num_heads: int) -> nn.Module:
        """Create Longformer-style attention (simplified)."""
        return self._make_vanilla_attention(hidden_size, num_heads)
    
    def _make_performer_attention(self,
                                hidden_size: int,
                                num_heads: int) -> nn.Module:
        """Create Performer-style attention (simplified)."""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def _make_vanilla_ffn(self,
                         hidden_size: int,
                         ffn_size: int) -> nn.Module:
        """Create vanilla feed-forward network."""
        return nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.ReLU(),
            nn.Linear(ffn_size, hidden_size)
        )
    
    def _make_gated_ffn(self,
                       hidden_size: int,
                       ffn_size: int) -> nn.Module:
        """Create gated feed-forward network."""
        return nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_size)
        )
    
    def forward(self,
               x: torch.Tensor,
               architecture: TransformerArchitecture) -> torch.Tensor:
        """Forward pass with given architecture."""
        # Apply position encoding
        x = self.position_encodings[architecture.position_encoding](x)
        
        # Process layers
        for i in range(architecture.num_layers):
            # Pre-norm if specified
            if architecture.norm_position in ['pre', 'sandwich']:
                x = nn.LayerNorm(architecture.hidden_size)(x)
            
            # Attention
            attention = self.attention_ops[architecture.attention_types[i]](
                architecture.hidden_size,
                architecture.num_heads[i]
            )
            x = x + attention(x, x, x)[0]
            
            # Post-norm if specified
            if architecture.norm_position in ['post', 'sandwich']:
                x = nn.LayerNorm(architecture.hidden_size)(x)
            
            # Feed-forward
            ffn_size = int(architecture.hidden_size * architecture.ffn_ratios[i])
            ffn = self.ffn_ops[architecture.ffn_types[i]](
                architecture.hidden_size,
                ffn_size
            )
            x = x + ffn(x)
            
            # Final norm if post-norm
            if architecture.norm_position == 'post':
                x = nn.LayerNorm(architecture.hidden_size)(x)
        
        return x

def search_transformer_architecture(
    model: TransformerNet,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_iterations: int = 1000,
    max_sequence_length: int = 512
) -> Tuple[TransformerArchitecture, float]:
    """Perform architecture search for transformer."""
    search_space = TransformerSearchSpace()
    nas = TransformerNAS(search_space)
    
    best_architecture = None
    best_perplexity = float('inf')
    
    device = next(model.parameters()).device
    
    for i in range(num_iterations):
        if i % 100 == 0:
            print(f"Search iteration {i}/{num_iterations}")
        
        # Generate or mutate architecture
        if best_architecture is None:
            architecture = nas.generate_random_architecture()
        else:
            architecture = nas.mutate_architecture(best_architecture)
        
        # Check FLOPs constraint
        flops = architecture.compute_flops(max_sequence_length)
        if flops > 1e12:  # 1 TFLOPs limit
            continue
        
        # Evaluate architecture
        perplexity = evaluate_architecture(
            model, architecture, valid_loader, device)
        
        if perplexity < best_perplexity:
            best_architecture = architecture
            best_perplexity = perplexity
            
            print(f"\nNew best architecture found!")
            print(f"Perplexity: {best_perplexity:.4f}")
            print(f"FLOPs: {flops:,}")
    
    return best_architecture, best_perplexity

def evaluate_architecture(
    model: TransformerNet,
    architecture: TransformerArchitecture,
    valid_loader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate transformer architecture using perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, architecture)
            loss = F.cross_entropy(
                outputs.view(-1, model.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            
            total_loss += loss.item() * (labels != -100).sum().item()
            total_tokens += (labels != -100).sum().item()
    
    return torch.exp(torch.tensor(total_loss / total_tokens))

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = TransformerNet(vocab_size=32000).to(device)
    
    # Create dummy data loaders
    # In practice, use real text dataset
    train_loader = DataLoader([], batch_size=32)
    valid_loader = DataLoader([], batch_size=32)
    
    # Perform search
    print("Starting transformer architecture search...")
    best_arch, best_perplexity = search_transformer_architecture(
        model, train_loader, valid_loader)
    
    print("\nSearch completed!")
    print(f"Best architecture:")
    print(f"Number of layers: {best_arch.num_layers}")
    print(f"Hidden size: {best_arch.hidden_size}")
    print(f"Number of heads: {best_arch.num_heads}")
    print(f"Attention types: {best_arch.attention_types}")
    print(f"FFN types: {best_arch.ffn_types}")
    print(f"FFN ratios: {best_arch.ffn_ratios}")
    print(f"Position encoding: {best_arch.position_encoding}")
    print(f"Norm position: {best_arch.norm_position}")
    print(f"Perplexity: {best_perplexity:.4f}")

if __name__ == "__main__":
    main()
