# Recipe 2.5: Once-for-All Networks

This recipe implements Once-for-All (OFA) Networks. Instead of searching for a single architecture, we train a single network that supports multiple sub-networks with different configurations.

## What You'll Learn
- Progressive shrinking training
- Elastic architecture design
- Knowledge distillation
- Dynamic architecture adaptation

## Key Components
1. Elastic Operations: Width, depth, kernel size
2. Progressive Training: From largest to smallest
3. Knowledge Distillation: Teacher-student training
4. Architecture Sampling: Sub-network extraction
