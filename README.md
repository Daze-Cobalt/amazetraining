# AmazeTraining

AmazeTraining is a modular Python maze-solving project that combines
traditional algorithms, procedural generation, and reinforcement learning.

The project demonstrates:
- Object-oriented programming
- Data structures and algorithms
- Procedural content generation
- Reinforcement learning
- File I/O and persistent model storage
- Unit testing
- Modular software architecture

---

# Features

## Maze Generation
- Random solvable maze generation
- Adjustable wall density
- Multiple difficulty levels
- Guaranteed solvable layouts

## Solvers
The project supports multiple maze-solving approaches:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Heuristic-based solving
- Reinforcement Learning agent

## Reinforcement Learning
The RL agent uses:
- Q-Learning concepts
- Neural-network-based Q-value prediction
- Encoded game states
- Epsilon-greedy exploration strategy

The trained neural network is saved and reloaded automatically using PyTorch.

## Visualization
- Interactive pygame visualization
- Real-time agent movement
- Replay system
- Keyboard controls

---

# Project Structure

```text
amazetraining/
│
├── data/
│   ├── datasets/
│   ├── levels/
│   ├── rl_logs/
│   └── rl_models/
│
├── rl_agent/
│   ├── agent.py
│   ├── q_network.py
│   ├── state_encoder.py
│   ├── strategy.py
│   ├── train.py
│   └── evaluate.py
│
├── src/
│   └── amaze_ai/
│
├── scripts/
│   ├── play_pygame.py
│   ├── solve_level.py
│   └── generate_levels.py
│
└── tests/