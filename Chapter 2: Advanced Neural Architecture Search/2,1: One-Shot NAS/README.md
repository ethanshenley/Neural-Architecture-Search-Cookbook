# Recipe 2.1: One-Shot Neural Architecture Search

This recipe implements One-Shot Neural Architecture Search using a weight-sharing supernet. Instead of training each architecture from scratch, we train a single supernet that contains all possible architectures.

## What You'll Learn
- Supernet design and training
- Path sampling strategies
- Uniform path sampling
- Architecture evaluation with shared weights
- Fairness in weight sharing

## Key Components
1. Supernet: Contains all possible architectures
2. Path Sampling: Uniform and fairness-aware sampling
3. Architecture Search: Efficient evaluation using shared weights
4. Gradient Management: Handling partial architecture updates
