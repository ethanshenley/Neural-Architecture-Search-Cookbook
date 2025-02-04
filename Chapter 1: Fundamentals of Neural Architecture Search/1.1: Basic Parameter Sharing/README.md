# Recipe 1.1: Basic Parameter Sharing in NAS

This recipe introduces the fundamental concept of parameter sharing in Neural Architecture Search (NAS). Instead of training each candidate architecture from scratch, we'll learn how to share weights across different architectures to make the search process more efficient.

## What You'll Learn
- Basic parameter sharing concept
- Simple architecture controller
- Training loop with shared weights
- Basic search space design

## Prerequisites
- PyTorch basics
- Understanding of CNNs

## Key Components
1. SharedCNN: A network with shared parameters
2. Controller: Samples architecture choices
3. Training loop: Alternates between updating shared weights and controller