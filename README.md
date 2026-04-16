
Caleb Governale

Python Setup: VS Code, venv, pygame

# Amaze

Amaze is a Python project that generates, validates, plays, and solves slide-based grid puzzles inspired by Amaze-style games. The player moves in one direction at a time and continues sliding until hitting a wall or boundary. The objective is to cover all valid tiles on the board.

This project includes:

- random level generation
- solvability validation
- breadth-first search (BFS) solving
- Pygame-based manual play
- test coverage for environment logic, solver logic, and generator behavior

## Features

- Grid-based puzzle environment with blocked and open tiles
- Random level generation with connectivity checks
- Solvable-only level generation using BFS validation
- Manual play through a Pygame GUI
- BFS-based automatic solver
- Dynamic board scaling and centering in the game window
- Early support for difficulty-filtered generation
- Unit tests for key systems

## How the Puzzle Works

- The board consists of open and blocked cells.
- The player starts on a valid tile.
- Each move continues in the chosen direction until the player reaches a wall or edge.
- Every traversed valid tile is marked as covered.
- The puzzle is solved when all valid tiles have been covered.

## Project Structure


amazetraining/
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── data/
│   ├── datasets/
│   ├── levels/
│   │   └── test_level.json
│   └── models/
├── scripts/
│   ├── generate_levels.py
│   ├── play_level.py
│   ├── play_pygame.py
│   ├── solve_level.py
│   └── train_policy.py
├── src/
│   └── amaze_ai/
│       ├── __init__.py
│       ├── config.py
│       ├── env.py
│       ├── generator.py
│       ├── heuristics.py
│       ├── io.py
│       ├── renderer.py
│       ├── solver.py
│       ├── types.py
│       └── utils.py
├── tests/
│   ├── test_env.py
│   ├── test_generator.py
│   └── test_solver.py
├── .env
├── pyproject.toml
├── pytest.ini
├── README.md
└── requirements.txt
