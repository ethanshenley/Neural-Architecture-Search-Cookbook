import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import zscore

@dataclass
class Metric:
    """Base metric class."""
    name: str
    weight: float
    minimize: bool = True
    history: List[float] = None
    
    def __post_init__(self):
        self.history = []
    
    def normalize(self, value: float) -> float:
        """Normalize metric value using z-score normalization."""
        if not self.history:
            normalized = 0.0
        else:
            normalized = zscore([*self.history, value])[-1]
        
        self.history.append(value)
        return -normalized if self.minimize else normalized

class CompositeObjective:
    """Combines multiple metrics into a single objective."""
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
        self.normalize_weights()
    
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total_weight = sum(m.weight for m in self.metrics)
        for metric in self.metrics:
            metric.weight /= total_weight
    
    def evaluate(self, architecture) -> Tuple[float, Dict[str, float]]:
        """Evaluate architecture using composite objective."""
        raw_values = {}
        normalized_scores = {}
        
        # Compute raw values and normalize
        for metric in self.metrics:
            raw_value = self._compute_metric(architecture, metric.name)
            raw_values[metric.name] = raw_value
            normalized_scores[metric.name] = metric.normalize(raw_value)
        
        # Compute weighted sum
        composite_score = sum(
            metric.weight * normalized_scores[metric.name]
            for metric in self.metrics
        )
        
        return composite_score, raw_values
    
    def _compute_metric(self, architecture, metric_name: str) -> float:
        """Compute individual metric value."""
        if metric_name == "accuracy":
            return architecture.evaluate_accuracy()
        elif metric_name == "latency":
            return architecture.compute_latency()
        elif metric_name == "flops":
            return architecture.compute_flops()
        elif metric_name == "params":
            return architecture.compute_params()
        elif metric_name == "memory":
            return architecture.compute_memory()
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

class MetricTracker:
    """Tracks metric statistics and correlations."""
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
        self.history: List[Dict[str, float]] = []
    
    def update(self, values: Dict[str, float]) -> None:
        """Update metric history."""
        self.history.append(values)
    
    def get_correlations(self) -> Dict[Tuple[str, str], float]:
        """Compute correlations between metrics."""
        correlations = {}
        values = np.array([[h[m.name] for m in self.metrics] 
                          for h in self.history])
        
        for i, metric1 in enumerate(self.metrics):
            for j, metric2 in enumerate(self.metrics):
                if i < j:
                    corr = np.corrcoef(values[:, i], values[:, j])[0, 1]
                    correlations[(metric1.name, metric2.name)] = corr
        
        return correlations
    
    def update_weights(self) -> None:
        """Update metric weights based on correlations."""
        if len(self.history) < 2:
            return
        
        correlations = self.get_correlations()
        
        # Reduce weights of highly correlated metrics
        for (m1, m2), corr in correlations.items():
            if abs(corr) > 0.8:  # High correlation threshold
                metric1 = next(m for m in self.metrics if m.name == m1)
                metric2 = next(m for m in self.metrics if m.name == m2)
                
                # Reduce weight of less important metric
                if metric1.weight < metric2.weight:
                    metric1.weight *= 0.9
                else:
                    metric2.weight *= 0.9

class CompositeNAS:
    """Neural Architecture Search with composite objectives."""
    def __init__(self,
                 composite_objective: CompositeObjective,
                 population_size: int = 100,
                 num_generations: int = 50,
                 mutation_prob: float = 0.1):
        self.objective = composite_objective
        self.tracker = MetricTracker(composite_objective.metrics)
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_prob = mutation_prob
        self.population: List[Dict] = []
        self.best_architecture = None
        self.best_score = float('-inf')
    
    def initialize_population(self, search_space) -> None:
        """Initialize random population."""
        self.population = []
        while len(self.population) < self.population_size:
            arch = search_space.random_architecture()
            score, values = self.objective.evaluate(arch)
            
            self.population.append({
                'architecture': arch,
                'score': score,
                'values': values
            })
            
            self.tracker.update(values)
            
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = arch
    
    def select_parent(self) -> Dict:
        """Tournament selection based on composite score."""
        tournament_size = 3
        tournament = np.random.choice(
            self.population, size=tournament_size, replace=False)
        return max(tournament, key=lambda x: x['score'])
    
    def create_offspring(self, 
                        parent1: Dict,
                        parent2: Dict,
                        search_space) -> Dict:
        """Create offspring through crossover and mutation."""
        # Crossover
        child_arch = search_space.crossover(
            parent1['architecture'],
            parent2['architecture']
        )
        
        # Mutation
        if np.random.random() < self.mutation_prob:
            child_arch = search_space.mutate(child_arch)
        
        # Evaluate child
        score, values = self.objective.evaluate(child_arch)
        
        return {
            'architecture': child_arch,
            'score': score,
            'values': values
        }
    
    def search(self, search_space) -> Tuple[any, Dict[str, float]]:
        """Perform architecture search."""
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
                self.tracker.update(child['values'])
            
            # Update metric weights
            self.tracker.update_weights()
            self.objective.normalize_weights()
            
            # Update population
            self.population = offspring
            
            # Update best architecture
            best_offspring = max(offspring, key=lambda x: x['score'])
            if best_offspring['score'] > self.best_score:
                self.best_score = best_offspring['score']
                self.best_architecture = best_offspring['architecture']
            
            # Print statistics
            self._print_generation_stats(generation)
        
        return self.best_architecture, self.objective.evaluate(
            self.best_architecture)[1]
    
    def _print_generation_stats(self, generation: int) -> None:
        """Print statistics for current generation."""
        scores = [arch['score'] for arch in self.population]
        
        print(f"\nGeneration {generation + 1} statistics:")
        print(f"Average score: {np.mean(scores):.4f}")
        print(f"Best score: {np.max(scores):.4f}")
        
        print("\nMetric weights:")
        for metric in self.objective.metrics:
            print(f"{metric.name}: {metric.weight:.4f}")
        
        if generation > 0:
            print("\nMetric correlations:")
            correlations = self.tracker.get_correlations()
            for (m1, m2), corr in correlations.items():
                print(f"{m1} - {m2}: {corr:.4f}")
    
    def visualize_metrics(self) -> None:
        """Visualize metric distributions and relationships."""
        metrics = [m.name for m in self.objective.metrics]
        num_metrics = len(metrics)
        
        fig, axes = plt.subplots(num_metrics, num_metrics, 
                                figsize=(3*num_metrics, 3*num_metrics))
        
        values = {m: [h[m] for h in self.tracker.history] for m in metrics}
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                ax = axes[i, j]
                
                if i == j:
                    # Histogram on diagonal
                    ax.hist(values[metric1], bins=20)
                    ax.set_title(f"{metric1}\nWeight: {self.objective.metrics[i].weight:.2f}")
                else:
                    # Scatter plot off diagonal
                    ax.scatter(values[metric1], values[metric2], alpha=0.5)
                    ax.set_xlabel(metric1)
                    ax.set_ylabel(metric2)
        
        plt.tight_layout()
        plt.show()

def main():
    # Define metrics
    metrics = [
        Metric("accuracy", weight=0.4, minimize=False),
        Metric("latency", weight=0.2),
        Metric("flops", weight=0.2),
        Metric("memory", weight=0.1),
        Metric("params", weight=0.1)
    ]
    
    composite_objective = CompositeObjective(metrics)
    
    # Create search space (implementation depends on specific use case)
    search_space = None  # Replace with actual search space
    
    # Initialize CompositeNAS
    nas = CompositeNAS(composite_objective)
    
    # Perform search
    best_architecture, best_values = nas.search(search_space)
    
    # Visualize results
    nas.visualize_metrics()
    
    # Print best architecture
    print("\nBest architecture found:")
    print("Metrics:")
    for name, value in best_values.items():
        print(f"{name}: {value:.4f}")
    print(f"Composite score: {nas.best_score:.4f}")

if __name__ == "__main__":
    main()
