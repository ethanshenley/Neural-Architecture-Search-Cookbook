import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from enum import Enum
import matplotlib.pyplot as plt
from collections import deque

class RuntimeCondition(Enum):
    """Different runtime conditions."""
    LOW_RESOURCE = "low_resource"
    NORMAL = "normal"
    HIGH_PERFORMANCE = "high_performance"
    ENERGY_SAVING = "energy_saving"

@dataclass
class ResourceMetrics:
    """System resource metrics."""
    cpu_usage: float
    memory_usage: float
    battery_level: float
    temperature: float
    network_bandwidth: float

class RuntimeMonitor:
    """Monitors system resources and conditions."""
    def __init__(self,
                 history_size: int = 100,
                 cpu_threshold: float = 80.0,
                 memory_threshold: float = 80.0,
                 battery_threshold: float = 20.0,
                 temperature_threshold: float = 70.0):
        self.history_size = history_size
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.battery_threshold = battery_threshold
        self.temperature_threshold = temperature_threshold
        
        self.metrics_history = deque(maxlen=history_size)
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system metrics."""
        # In practice, this would use system APIs
        # Here we simulate the metrics
        return ResourceMetrics(
            cpu_usage=np.random.uniform(0, 100),
            memory_usage=np.random.uniform(0, 100),
            battery_level=np.random.uniform(0, 100),
            temperature=np.random.uniform(30, 90),
            network_bandwidth=np.random.uniform(0, 100)
        )
    
    def update(self) -> None:
        """Update metrics history."""
        current_metrics = self.get_current_metrics()
        self.metrics_history.append(current_metrics)
    
    def get_condition(self) -> RuntimeCondition:
        """Determine current runtime condition."""
        if not self.metrics_history:
            return RuntimeCondition.NORMAL
        
        current_metrics = self.metrics_history[-1]
        
        # Check for low resource condition
        if (current_metrics.cpu_usage > self.cpu_threshold or
            current_metrics.memory_usage > self.memory_threshold):
            return RuntimeCondition.LOW_RESOURCE
        
        # Check for energy saving condition
        if current_metrics.battery_level < self.battery_threshold:
            return RuntimeCondition.ENERGY_SAVING
        
        # Check for high performance condition
        if (current_metrics.cpu_usage < 50.0 and
            current_metrics.memory_usage < 50.0 and
            current_metrics.temperature < 60.0):
            return RuntimeCondition.HIGH_PERFORMANCE
        
        return RuntimeCondition.NORMAL

@dataclass
class ArchitectureConfig:
    """Configuration for a specific architecture."""
    name: str
    flops: int
    memory: int
    latency: float
    accuracy: float
    energy_efficiency: float

class AdaptiveArchitecture:
    """Architecture that can adapt to different conditions."""
    def __init__(self,
                 base_config: ArchitectureConfig,
                 efficient_config: ArchitectureConfig,
                 performance_config: ArchitectureConfig):
        self.configs = {
            RuntimeCondition.NORMAL: base_config,
            RuntimeCondition.LOW_RESOURCE: efficient_config,
            RuntimeCondition.HIGH_PERFORMANCE: performance_config,
            RuntimeCondition.ENERGY_SAVING: efficient_config
        }
        self.current_config = base_config
        self.switches = 0
        self.switch_history: List[Tuple[float, str]] = []
    
    def switch_config(self, condition: RuntimeCondition) -> None:
        """Switch to appropriate configuration."""
        new_config = self.configs[condition]
        if new_config != self.current_config:
            self.current_config = new_config
            self.switches += 1
            self.switch_history.append(
                (time.time(), new_config.name))
            print(f"Switching to {new_config.name} configuration")

class DynamicNAS:
    """Dynamic Neural Architecture Search."""
    def __init__(self,
                 monitor: RuntimeMonitor,
                 update_interval: float = 1.0,
                 stability_threshold: int = 5):
        self.monitor = monitor
        self.update_interval = update_interval
        self.stability_threshold = stability_threshold
        self.last_update = time.time()
        self.condition_history = deque(maxlen=stability_threshold)
        self.performance_history: Dict[str, List[float]] = {
            'latency': [],
            'accuracy': [],
            'energy': []
        }
    
    def search_base_architectures(self, search_space) -> Tuple[
            ArchitectureConfig, ArchitectureConfig, ArchitectureConfig]:
        """Search for base architectures for different conditions."""
        # In practice, this would use a full NAS algorithm
        # Here we simulate the search
        base_config = ArchitectureConfig(
            name="base",
            flops=1e9,
            memory=1e6,
            latency=50.0,
            accuracy=0.9,
            energy_efficiency=0.7
        )
        
        efficient_config = ArchitectureConfig(
            name="efficient",
            flops=5e8,
            memory=5e5,
            latency=30.0,
            accuracy=0.85,
            energy_efficiency=0.9
        )
        
        performance_config = ArchitectureConfig(
            name="performance",
            flops=2e9,
            memory=2e6,
            latency=70.0,
            accuracy=0.95,
            energy_efficiency=0.5
        )
        
        return base_config, efficient_config, performance_config
    
    def should_switch(self, current_condition: RuntimeCondition) -> bool:
        """Determine if architecture should switch based on stability."""
        self.condition_history.append(current_condition)
        
        if len(self.condition_history) < self.stability_threshold:
            return False
        
        # Check if condition has been stable
        return all(c == current_condition 
                  for c in self.condition_history)
    
    def update_performance_metrics(self, 
                                 architecture: AdaptiveArchitecture) -> None:
        """Update performance history."""
        config = architecture.current_config
        self.performance_history['latency'].append(config.latency)
        self.performance_history['accuracy'].append(config.accuracy)
        self.performance_history['energy'].append(config.energy_efficiency)
    
    def run(self, 
            architecture: AdaptiveArchitecture, 
            duration: float) -> None:
        """Run dynamic adaptation for specified duration."""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Check if it's time to update
            if current_time - self.last_update >= self.update_interval:
                # Update system metrics
                self.monitor.update()
                
                # Get current condition
                condition = self.monitor.get_condition()
                
                # Check if we should switch
                if self.should_switch(condition):
                    architecture.switch_config(condition)
                
                # Update performance metrics
                self.update_performance_metrics(architecture)
                
                self.last_update = current_time
            
            # Simulate some work
            time.sleep(0.1)
    
    def visualize_performance(self) -> None:
        """Visualize performance metrics over time."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot latency
        ax1.plot(self.performance_history['latency'])
        ax1.set_title('Latency over time')
        ax1.set_ylabel('Latency (ms)')
        
        # Plot accuracy
        ax2.plot(self.performance_history['accuracy'])
        ax2.set_title('Accuracy over time')
        ax2.set_ylabel('Accuracy')
        
        # Plot energy efficiency
        ax3.plot(self.performance_history['energy'])
        ax3.set_title('Energy Efficiency over time')
        ax3.set_ylabel('Efficiency')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_switches(self, architecture: AdaptiveArchitecture) -> None:
        """Visualize architecture switches over time."""
        if not architecture.switch_history:
            print("No switches recorded")
            return
        
        times, configs = zip(*architecture.switch_history)
        times = [t - times[0] for t in times]  # Relative times
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, configs, 'ro-')
        plt.title('Architecture Switches Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Configuration')
        plt.grid(True)
        plt.show()

def main():
    # Create runtime monitor
    monitor = RuntimeMonitor()
    
    # Create dynamic NAS
    nas = DynamicNAS(monitor)
    
    # Search for base architectures
    base_config, efficient_config, performance_config = \
        nas.search_base_architectures(None)  # Replace with actual search space
    
    # Create adaptive architecture
    architecture = AdaptiveArchitecture(
        base_config, efficient_config, performance_config)
    
    # Run dynamic adaptation
    print("Starting dynamic adaptation...")
    nas.run(architecture, duration=30.0)
    
    # Print statistics
    print("\nAdaptation Statistics:")
    print(f"Total switches: {architecture.switches}")
    print("\nFinal Configuration:")
    print(f"Name: {architecture.current_config.name}")
    print(f"FLOPs: {architecture.current_config.flops:,}")
    print(f"Memory: {architecture.current_config.memory:,}")
    print(f"Latency: {architecture.current_config.latency:.2f} ms")
    print(f"Accuracy: {architecture.current_config.accuracy:.4f}")
    print(f"Energy Efficiency: {architecture.current_config.energy_efficiency:.4f}")
    
    # Visualize results
    nas.visualize_performance()
    nas.visualize_switches(architecture)

if __name__ == "__main__":
    main()
