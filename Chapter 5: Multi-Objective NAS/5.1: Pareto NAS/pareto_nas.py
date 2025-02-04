import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict

@dataclass
class Objective:
    """Represents a single objective to optimize."""
    name: str
    minimize: bool = True  # True if we want to minimize this objective
    
    def __call__(self, architecture) -> float:
        """Calculate objective value for an architecture."""
        raise NotImplementedError

class AccuracyObjective(Objective):
    """Accuracy objective (maximize)."""
    def __init__(self):
        super().__init__("accuracy", minimize=False)
    
    def __call__(self, architecture) -> float:
        return architecture.evaluate_accuracy()

class LatencyObjective(Objective):
    """Latency objective (minimize)."""
    def __init__(self):
        super().__init__("latency", minimize=True)
    
    def __call__(self, architecture) -> float:
        return architecture.compute_latency()

class FLOPsObjective(Objective):
    """FLOPs objective (minimize)."""
    def __init__(self):
        super().__init__("flops", minimize=True)
    
    def __call__(self, architecture) -> float:
        return architecture.compute_flops()

class MemoryObjective(Objective):
    """Memory objective (minimize)."""
    def __init__(self):
        super().__init__("memory", minimize=True)
    
    def __call__(self, architecture) -> float:
        return architecture.compute_memory()

@dataclass
class Solution:
    """Represents a solution (architecture) with its objective values."""
    architecture: any
    objectives: Dict[str, float]
    rank: int = 0
    crowding_distance: float = 0.0

class ParetoNAS:
    """Pareto-optimal Neural Architecture Search."""
    def __init__(self,
                 objectives: List[Objective],
                 population_size: int = 100,
                 num_generations: int = 50):
        self.objectives = objectives
        self.population_size = population_size
        self.num_generations = num_generations
        self.population: List[Solution] = []
        self.pareto_fronts: List[List[Solution]] = []
    
    def initialize_population(self, search_space) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            arch = search_space.random_architecture()
            obj_values = {obj.name: obj(arch) for obj in self.objectives}
            self.population.append(Solution(arch, obj_values))
    
    def dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """Check if solution 1 dominates solution 2."""
        at_least_one_better = False
        for obj in self.objectives:
            val1 = sol1.objectives[obj.name]
            val2 = sol2.objectives[obj.name]
            
            if obj.minimize:
                if val1 > val2:  # sol1 is worse
                    return False
                if val1 < val2:  # sol1 is better
                    at_least_one_better = True
            else:
                if val1 < val2:  # sol1 is worse
                    return False
                if val1 > val2:  # sol1 is better
                    at_least_one_better = True
        
        return at_least_one_better
    
    def fast_non_dominated_sort(self) -> List[List[Solution]]:
        """Implement NSGA-II's fast non-dominated sorting."""
        fronts = [[]]
        for p in self.population:
            p.domination_count = 0
            p.dominated_solutions = []
            
            for q in self.population:
                if self.dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self.dominates(q, p):
                    p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def calculate_crowding_distance(self, front: List[Solution]) -> None:
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            for solution in front:
                solution.crowding_distance = float('inf')
            return
        
        for solution in front:
            solution.crowding_distance = 0
        
        for obj in self.objectives:
            front.sort(key=lambda x: x.objectives[obj.name])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            obj_range = (front[-1].objectives[obj.name] - 
                        front[0].objectives[obj.name])
            if obj_range == 0:
                continue
            
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives[obj.name] -
                    front[i - 1].objectives[obj.name]
                ) / obj_range
    
    def crowding_operator(self, sol1: Solution, sol2: Solution) -> bool:
        """Compare solutions based on rank and crowding distance."""
        if sol1.rank < sol2.rank:
            return True
        if sol1.rank > sol2.rank:
            return False
        return sol1.crowding_distance > sol2.crowding_distance
    
    def select_parents(self) -> Tuple[Solution, Solution]:
        """Select parents using binary tournament selection."""
        candidates = np.random.choice(
            len(self.population), size=4, replace=False)
        tournament1 = [self.population[candidates[0]], 
                      self.population[candidates[1]]]
        tournament2 = [self.population[candidates[2]], 
                      self.population[candidates[3]]]
        
        parent1 = max(tournament1, 
                     key=lambda x: (x.rank, x.crowding_distance))
        parent2 = max(tournament2, 
                     key=lambda x: (x.rank, x.crowding_distance))
        
        return parent1, parent2
    
    def create_offspring(self, 
                        parent1: Solution, 
                        parent2: Solution,
                        search_space) -> Solution:
        """Create offspring through crossover and mutation."""
        # Crossover
        child_arch = search_space.crossover(
            parent1.architecture, parent2.architecture)
        
        # Mutation
        child_arch = search_space.mutate(child_arch)
        
        # Evaluate objectives
        obj_values = {obj.name: obj(child_arch) for obj in self.objectives}
        
        return Solution(child_arch, obj_values)
    
    def search(self, search_space) -> List[Solution]:
        """Perform Pareto-optimal architecture search."""
        # Initialize population
        self.initialize_population(search_space)
        
        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}/{self.num_generations}")
            
            # Create offspring population
            offspring = []
            while len(offspring) < self.population_size:
                parent1, parent2 = self.select_parents()
                child = self.create_offspring(parent1, parent2, search_space)
                offspring.append(child)
            
            # Combine parent and offspring populations
            combined_pop = self.population + offspring
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort()
            
            # Calculate crowding distance for each front
            for front in fronts:
                self.calculate_crowding_distance(front)
            
            # Select next generation
            next_population = []
            front_idx = 0
            while len(next_population) + len(fronts[front_idx]) <= self.population_size:
                next_population.extend(fronts[front_idx])
                front_idx += 1
            
            # Fill remaining slots using crowding distance
            if len(next_population) < self.population_size:
                fronts[front_idx].sort(
                    key=lambda x: x.crowding_distance, reverse=True)
                next_population.extend(
                    fronts[front_idx][:self.population_size - len(next_population)])
            
            self.population = next_population
            
            # Store Pareto front
            self.pareto_fronts.append(fronts[0])
            
            # Print statistics
            self._print_generation_stats(generation)
        
        return self.pareto_fronts[-1]
    
    def _print_generation_stats(self, generation: int) -> None:
        """Print statistics for current generation."""
        stats = defaultdict(list)
        for solution in self.population:
            for obj_name, value in solution.objectives.items():
                stats[obj_name].append(value)
        
        print(f"\nGeneration {generation + 1} statistics:")
        for obj_name, values in stats.items():
            print(f"{obj_name}:")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Best: {np.max(values) if not self.objectives[0].minimize else np.min(values):.4f}")
    
    def visualize_pareto_front(self, 
                             obj1_idx: int = 0, 
                             obj2_idx: int = 1) -> None:
        """Visualize 2D Pareto front."""
        obj1_name = self.objectives[obj1_idx].name
        obj2_name = self.objectives[obj2_idx].name
        
        plt.figure(figsize=(10, 6))
        
        # Plot all solutions
        x = [sol.objectives[obj1_name] for sol in self.population]
        y = [sol.objectives[obj2_name] for sol in self.population]
        plt.scatter(x, y, c='blue', alpha=0.5, label='All solutions')
        
        # Plot Pareto front
        pareto_front = self.pareto_fronts[-1]
        x = [sol.objectives[obj1_name] for sol in pareto_front]
        y = [sol.objectives[obj2_name] for sol in pareto_front]
        plt.scatter(x, y, c='red', label='Pareto front')
        
        # Connect Pareto front points
        points = list(zip(x, y))
        points.sort()
        x, y = zip(*points)
        plt.plot(x, y, 'r--')
        
        plt.xlabel(obj1_name)
        plt.ylabel(obj2_name)
        plt.title('Pareto Front Visualization')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # Define objectives
    objectives = [
        AccuracyObjective(),
        LatencyObjective(),
        FLOPsObjective(),
        MemoryObjective()
    ]
    
    # Create search space (implementation depends on specific use case)
    search_space = None  # Replace with actual search space (feel free to use one of our many examples)
    
    # Initialize ParetoNAS
    nas = ParetoNAS(objectives)
    
    # Perform search
    pareto_front = nas.search(search_space)
    
    # Visualize results
    nas.visualize_pareto_front()
    
    # Print Pareto-optimal solutions
    print("\nPareto-optimal solutions:")
    for i, solution in enumerate(pareto_front, 1):
        print(f"\nSolution {i}:")
        for obj_name, value in solution.objectives.items():
            print(f"{obj_name}: {value:.4f}")

if __name__ == "__main__":
    main()
