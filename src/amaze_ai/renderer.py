from amaze_ai.env import AmazeEnv
from amaze_ai.types import State

def render_ascii(env: AmazeEnv, state: State) -> str:
    lines = []

    for r in range(env.rows):
        row_chars = []
        for c in range(env.cols):
            pos = (r, c)

            if env.grid[r][c] == 0:
                row_chars.append('#')
            elif pos == state.player:
                row_chars.append('P')
            else:
                bit = env.cell_to_bit[pos]
                painted = (state.painted_mask >> bit) & 1
                row_chars.append('.' if painted else '_')

        lines.append(' '.join(row_chars))

    return "\n".join(lines)