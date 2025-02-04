# Recipe 1.3: Policy Gradient NAS

This recipe implements Neural Architecture Search using policy gradients (REINFORCE algorithm). The controller learns to generate better architectures by maximizing expected reward through policy gradient updates.

## What You'll Learn
- Policy gradient optimization for NAS
- REINFORCE algorithm implementation
- Reward function design
- Baseline strategies for variance reduction

## Key Components
1. Policy Network (Controller): Samples architecture decisions
2. REINFORCE Training: Updates controller using policy gradients
3. Reward Design: Architecture evaluation and reward calculation
4. Baseline: Moving average for variance reduction
