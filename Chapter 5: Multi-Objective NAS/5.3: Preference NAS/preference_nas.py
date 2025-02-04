import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import norm

@dataclass
class Preference:
    """Represents a user preference for an objective."""
    name: str
    target: float  # Target value
    tolerance: float  # Acceptable deviation
    priority: float  # Importance weight (0-1)
    
    def compute_satisfaction(self, value: float) -> float:
        """Compute how well a value satisfies the preference."""
        # Use normal distribution to model satisfaction
        z_score = abs(value - self.target) / self.tolerance
        satisfaction = norm.pdf(z_score) / norm.pdf(0)
        return satisfaction * self.priority

class PreferenceSet:
    """Collection of preferences with evaluation methods."""
    def __init__(self, preferences: List[Preference]):
        self.preferences = preferences
        self.normalize_priorities()
    
    def normalize_priorities(self) -> None:
        """Normalize priority weights to sum to 1."""
        total_priority = sum(p.priority for p in self.preferences)
        for pref in self.preferences:
            pref.priority /= total_priority
    
    def evaluate(self, metrics: Dict[str, float]) -> float:
        """Compute overall preference satisfaction score."""
        total_satisfaction = 0.0
        for preference in self.preferences:
            satisfaction = preference.compute_satisfaction(
                metrics[preference.name])
            total_satisfaction += satisfaction
        return total_satisfaction

class PreferenceModel:
    """Model for learning and updating preferences."""
    def __init__(self, preference_set: PreferenceSet):
        self.preference_set = preference_set
        self.history: List[Tuple[Dict[str, float], float]] = []
    
    def update(self, metrics: Dict[str, float], user_rating: float) -> None:
        """Update preference model with new feedback."""
        self.history.append((metrics, user_rating))
        self._adjust_preferences()
    
    def _adjust_preferences(self) -> None:
        """Adjust preferences based on feedback history."""
        if len(self.history) < 2:
            return
        
        # Analyze recent feedback
        recent_metrics, recent_rating = self.history[-1]
        
        # Adjust tolerances and priorities based on feedback
        for preference in self.preference_set.preferences:
            values = [h[0][preference.name] for h in self.history]
            ratings = [h[1] for h in self.history]
            
            # Compute correlation between metric and ratings
            correlation = np.corrcoef(values, ratings)[0, 1]
            
            # Adjust priority based on correlation
            preference.priority *= (1 + 0.1 * correlation)
            
            # Adjust tolerance based on value distribution
            std_dev = np.std(values)
            preference.tolerance = (preference.tolerance + std_dev) / 2
        
        # Renormalize priorities
        self.preference_set.normalize_priorities()

class PreferenceNAS:
    """Preference-based Neural Architecture Search."""
    def __init__(self,
                 preference_set: PreferenceSet,
                 population_size: int = 100,
                 num_generations: int = 50,
                 exploration_rate: float = 0.2):
        self.preference_set = preference_set
        self.preference_model = PreferenceModel(preference_set)
        self.population_size = population_size
        self.num_generations = num_generations
        self.exploration_rate = exploration_rate
        self.population: List[Dict] = []  # List of (architecture, metrics)
        self.best_architecture = None
        self.best_satisfaction = 0.0
    
    def initialize_population(self, search_space) -> None:
        """Initialize random population."""
        self.population = []
        while len(self.population) < self.population_size:
            arch = search_space.random_architecture()
            metrics = self._evaluate_architecture(arch)
            satisfaction = self.preference_set.evaluate(metrics)
            
            self.population.append({
                'architecture': arch,
                'metrics': metrics,
                'satisfaction': satisfaction
            })
            
            if satisfaction > self.best_satisfaction:
                self.best_satisfaction = satisfaction
                self.best_architecture = arch
    
    def _evaluate_architecture(self, architecture) -> Dict[str, float]:
        """Evaluate architecture and compute metrics."""
        return {
            'accuracy': architecture.evaluate_accuracy(),
            'latency': architecture.compute_latency(),
            'memory': architecture.compute_memory(),
            'flops': architecture.compute_flops()
        }
    
    def select_parent(self) -> Dict:
        """Tournament selection based on preference satisfaction."""
        tournament_size = 3
        tournament = np.random.choice(
            self.population, size=tournament_size, replace=False)
        return max(tournament, key=lambda x: x['satisfaction'])
    
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
        
        # Mutation with exploration
        if np.random.random() < self.exploration_rate:
            child_arch = search_space.mutate(child_arch)
        
        # Evaluate child
        metrics = self._evaluate_architecture(child_arch)
        satisfaction = self.preference_set.evaluate(metrics)
        
        return {
            'architecture': child_arch,
            'metrics': metrics,
            'satisfaction': satisfaction
        }
    
    def get_user_feedback(self, architecture: Dict) -> float:
        """Simulate or get real user feedback."""
        # In practice, this would involve real user interaction
        # Here we simulate it based on preference satisfaction
        return architecture['satisfaction']
    
    def search(self, search_space) -> Tuple[any, Dict[str, float]]:
        """Perform preference-based architecture search."""
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
            
            # Get feedback for some offspring
            feedback_samples = np.random.choice(
                offspring, size=5, replace=False)
            for sample in feedback_samples:
                feedback = self.get_user_feedback(sample)
                self.preference_model.update(sample['metrics'], feedback)
            
            # Update population
            self.population = offspring
            
            # Update best architecture
            best_offspring = max(offspring, key=lambda x: x['satisfaction'])
            if best_offspring['satisfaction'] > self.best_satisfaction:
                self.best_satisfaction = best_offspring['satisfaction']
                self.best_architecture = best_offspring['architecture']
            
            # Print statistics
            self._print_generation_stats(generation)
        
        return self.best_architecture, self._evaluate_architecture(
            self.best_architecture)
    
    def _print_generation_stats(self, generation: int) -> None:
        """Print statistics for current generation."""
        satisfactions = [arch['satisfaction'] for arch in self.population]
        
        print(f"\nGeneration {generation + 1} statistics:")
        print(f"Average satisfaction: {np.mean(satisfactions):.4f}")
        print(f"Best satisfaction: {np.max(satisfactions):.4f}")
        
        print("\nCurrent preferences:")
        for pref in self.preference_set.preferences:
            print(f"{pref.name}:")
            print(f"  Target: {pref.target:.4f}")
            print(f"  Tolerance: {pref.tolerance:.4f}")
            print(f"  Priority: {pref.priority:.4f}")
    
    def visualize_preferences(self) -> None:
        """Visualize preference satisfaction distribution."""
        metrics = list(self.population[0]['metrics'].keys())
        num_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 4))
        
        for i, metric in enumerate(metrics):
            values = [arch['metrics'][metric] for arch in self.population]
            satisfaction = [arch['satisfaction'] for arch in self.population]
            
            # Create scatter plot
            scatter = axes[i].scatter(values, satisfaction, 
                                    alpha=0.5, c=satisfaction, cmap='viridis')
            
            # Plot preference target and tolerance
            pref = next(p for p in self.preference_set.preferences 
                       if p.name == metric)
            axes[i].axvline(x=pref.target, color='r', linestyle='--', 
                          label='Target')
            axes[i].axvspan(pref.target - pref.tolerance, 
                          pref.target + pref.tolerance,
                          alpha=0.2, color='r', label='Tolerance')
            
            axes[i].set_title(f"{metric}\nPriority: {pref.priority:.2f}")
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Satisfaction')
            axes[i].legend()
        
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.show()

def main():
    # Define preferences
    preferences = [
        Preference("accuracy", target=0.95, tolerance=0.02, priority=0.4),
        Preference("latency", target=50.0, tolerance=10.0, priority=0.3),
        Preference("memory", target=2.0, tolerance=0.5, priority=0.2),
        Preference("flops", target=5e8, tolerance=1e8, priority=0.1)
    ]
    
    preference_set = PreferenceSet(preferences)
    
    # Create search space (implementation depends on specific use case)
    search_space = None  # Replace with actual search space
    
    # Initialize PreferenceNAS
    nas = PreferenceNAS(preference_set)
    
    # Perform search
    best_architecture, best_metrics = nas.search(search_space)
    
    # Visualize results
    nas.visualize_preferences()
    
    # Print best architecture
    print("\nBest architecture found:")
    print("Metrics:")
    for name, value in best_metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Satisfaction: {nas.best_satisfaction:.4f}")

if __name__ == "__main__":
    main()
