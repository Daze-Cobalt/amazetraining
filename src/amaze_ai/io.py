import json
from pathlib import Path
from typing import List, Tuple

Grid = List[List[int]]
Position = Tuple[int, int]


def save_level(path: str | Path, grid: Grid, start: Position) -> None:
    data = {
        "grid": grid,
        "start": list(start),
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_level(path: str | Path) -> tuple[Grid, Position]:
    data = json.loads(Path(path).read_text())
    grid = data["grid"]
    start = tuple(data["start"])
    return grid, start