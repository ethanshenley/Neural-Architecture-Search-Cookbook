import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt

class ConstraintType(Enum):
    """Types of constraints."""
    HARD = "hard"  # Must be satisfied
    SOFT = "soft"  # Preferred but not required

@dataclass
class Constraint:
    """Represents a constraint on the architecture."""
    name: str
    type: ConstraintType
    threshold: float
    weight: float = 1.0  # Weight for soft constraints
    
    def evaluate(self, value: float) -> Tuple[bool, float]:
        """Evaluate constraint satisfaction and violation."""
        satisfied = value <= self.threshold
        violation = max(0, value - self.threshold)
        return satisfied, violation

class ConstraintSet:
    """Collection of constraints with evaluation methods."""
    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints
        self.hard_constraints = [c for c in constraints 
                               if c.type == ConstraintType.HARD]
        self.soft_constraints = [c for c in constraints 
                               if c.type == ConstraintType.SOFT]
    
    def is_feasible(self, metrics: Dict[str, float]) -> bool:
        """Check if all hard constraints are satisfied."""
        for constraint in self.hard_constraints:
            satisfied, _ = constraint.evaluate(metrics[constraint.name])
            if not satisfied:
                return False
        return True
    
    def compute_penalty(self, metrics: Dict[str, float]) -> float:
        """Compute weighted sum of soft constraint violations."""
        total_penalty = 0.0
        for constraint in self.soft_constraints:
            _, violation = constraint.evaluate(metrics[constraint.name])
            total_penalty += constraint.weight * violation
        return total_penalty

class ConstrainedArchitecture:
    """Architecture with constraint-aware evaluation."""
    def __init__(self, architecture: any, constraint_set: ConstraintSet):
        self.architecture = architecture
        self.constraint_set = constraint_set
        self.metrics: Dict[str, float] = {}
        self.feasible: bool = False
        self.penalty: float = 0.0
        self.fitness: float = 0.0
    
    def evaluate(self) -> None:
        """Evaluate architecture and check constraints."""
        # Compute metrics
        self.metrics = {
            "latency": self.architecture.compute_latency(),
            "memory": self.architecture.compute_memory(),
            "flops": self.architecture.compute_flops(),
            "params": self.architecture.compute_params()
        }
        
        # Check feasibility
        self.feasible = self.constraint_set.is_feasible(self.metrics)
        
        # Compute penalty
        self.penalty = self.constraint_set.compute_penalty(self.metrics)
        
        # Compute fitness (accuracy - penalty)
        accuracy = self.architecture.evaluate_accuracy()
        self.fitness = accuracy - self.penalty if self.feasible else -float('inf')

class ConstrainedNAS:
    """Constrained Neural Architecture Search."""
    def __init__(self,
                 constraint_set: ConstraintSet,
                 population_size: int = 100,
                 num_generations: int = 50,
                 mutation_prob: float = 0.1):
        self.constraint_set = constraint_set
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_prob = mutation_prob
        self.population: List[ConstrainedArchitecture] = []
        self.best_feasible: Optional[ConstrainedArchitecture] = None
    
    def initialize_population(self, search_space) -> None:
        """Initialize random population."""
        self.population = []
        while len(self.population) < self.population_size:
            arch = search_space.random_architecture()
            constrained_arch = ConstrainedArchitecture(
                arch, self.constraint_set)
            constrained_arch.evaluate()
            
            # Only add if feasible (for initial population)
            if constrained_arch.feasible:
                self.population.append(constrained_arch)
                
                # Update best feasible
                if (self.best_feasible is None or 
                    constrained_arch.fitness > self.best_feasible.fitness):
                    self.best_feasible = constrained_arch
    
    def select_parent(self) -> ConstrainedArchitecture:
        """Tournament selection with feasibility rules."""
        tournament_size = 3
        candidates = np.random.choice(
            len(self.population), size=tournament_size, replace=False)
        
        tournament = [self.population[i] for i in candidates]
        
        # First priority: feasibility
        feasible = [arch for arch in tournament if arch.feasible]
        if feasible:
            return max(feasible, key=lambda x: x.fitness)
        
        # If no feasible solutions, minimize constraint violation
        return min(tournament, key=lambda x: x.penalty)
    
    def create_offspring(self, 
                        parent1: ConstrainedArchitecture,
                        parent2: ConstrainedArchitecture,
                        search_space) -> ConstrainedArchitecture:
        """Create offspring through crossover and mutation."""
        # Crossover
        child_arch = search_space.crossover(
            parent1.architecture, parent2.architecture)
        
        # Mutation
        if np.random.random() < self.mutation_prob:
            child_arch = search_space.mutate(child_arch)
        
        # Create and evaluate constrained architecture
        child = ConstrainedArchitecture(child_arch, self.constraint_set)
        child.evaluate()
        
        return child
    
    def search(self, search_space) -> ConstrainedArchitecture:
        """Perform constrained architecture search."""
        # Initialize population
        print("Initializing population...")
        self.initialize_population(search_space)
        
        for generation in range(self.num_generations):
            print(f"\nGeneration {generation + 1}/{self.num_generations}")
            
            # Create offspring population
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child = self.create_offspring(parent1, parent2, search_space)
                offspring.append(child)
            
            # Replace population with offspring
            self.population = offspring
            
            # Update best feasible solution
            for arch in self.population:
                if arch.feasible and (self.best_feasible is None or 
                                    arch.fitness > self.best_feasible.fitness):
                    self.best_feasible = arch
            
            # Print statistics
            self._print_generation_stats(generation)
        
        return self.best_feasible
    
    def _print_generation_stats(self, generation: int) -> None:
        """Print statistics for current generation."""
        feasible_count = sum(1 for arch in self.population if arch.feasible)
        feasible_archs = [arch for arch in self.population if arch.feasible]
        
        print(f"\nGeneration {generation + 1} statistics:")
        print(f"Feasible architectures: {feasible_count}/{self.population_size}")
        
        if feasible_archs:
            avg_fitness = np.mean([arch.fitness for arch in feasible_archs])
            best_fitness = max(arch.fitness for arch in feasible_archs)
            print(f"Average feasible fitness: {avg_fitness:.4f}")
            print(f"Best feasible fitness: {best_fitness:.4f}")
        
        if self.best_feasible:
            print("\nBest feasible solution metrics:")
            for name, value in self.best_feasible.metrics.items():
                print(f"{name}: {value:.4f}")
    
    def visualize_constraints(self) -> None:
        """Visualize constraint satisfaction across population."""
        metrics = list(self.population[0].metrics.keys())
        num_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 4))
        
        for i, metric in enumerate(metrics):
            values = [arch.metrics[metric] for arch in self.population]
            threshold = next((c.threshold for c in self.constraint_set.constraints 
                            if c.name == metric), None)
            
            axes[i].hist(values, bins=20)
            if threshold is not None:
                axes[i].axvline(x=threshold, color='r', linestyle='--', 
                              label='Constraint')
            
            axes[i].set_title(metric)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Count')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Define constraints
    constraints = [
        Constraint("latency", ConstraintType.HARD, threshold=100.0),  # ms
        Constraint("memory", ConstraintType.HARD, threshold=4.0),     # GB
        Constraint("flops", ConstraintType.SOFT, threshold=1e9, weight=0.1),
        Constraint("params", ConstraintType.SOFT, threshold=5e6, weight=0.05)
    ]
    
    constraint_set = ConstraintSet(constraints)
    
    # Create search space (implementation depends on specific use case)
    search_space = None  # Replace with actual search space
    
    # Initialize ConstrainedNAS
    nas = ConstrainedNAS(constraint_set)
    
    # Perform search
    best_architecture = nas.search(search_space)
    
    # Visualize results
    nas.visualize_constraints()
    
    # Print best architecture
    if best_architecture:
        print("\nBest feasible architecture found:")
        print(f"Fitness: {best_architecture.fitness:.4f}")
        print("\nMetrics:")
        for name, value in best_architecture.metrics.items():
            print(f"{name}: {value:.4f}")
    else:
        print("\nNo feasible architecture found!")

if __name__ == "__main__":
    main()
