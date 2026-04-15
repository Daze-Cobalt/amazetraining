from collections import deque
from typing import Dict, List, Optional, Tuple

from amaze_ai.config import ACTIONS
from amaze_ai.env import AmazeEnv
from amaze_ai.types import State

def solve_bfs(env: AmazeEnv) -> Optional[List[int]]:
    start = env.reset()

    queue = deque([start])
    visited = {start}
    parent: Dict[State, Tuple[Optional[State], Optional[int]]] = {
        start: (None, None)
    }

    while queue:
        current = queue.popleft()

        if env.is_goal(current):
            return _reconstruct_path(parent, current)
        
        for action in ACTIONS:
            result = env.step(current, action)
            nxt = result.state

            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = (current, action)
                queue.append(nxt)

    return None

def _reconstruct_path(
        parent: Dict[State, Tuple[Optional[State], Optional[int]]],
        goal_state: State
) -> List[int]:
    actions: List[int] = []
    cur = goal_state

    while parent[cur][0] is not None:
        prev, action = parent[cur]
        assert action is not None
        actions.append(action)
        assert prev is not None
        cur = prev

    actions.reverse()
    return actions