import sys
import pygame

from amaze_ai.config import UP, DOWN, LEFT, RIGHT, ACTION_NAMES
from amaze_ai.env import AmazeEnv
from amaze_ai.generator import (
    make_random_solvable_level,
    make_easy_level,
    make_medium_level,
    make_hard_level,
)
from amaze_ai.solver import solve_bfs

from rl_agent.agent import QLearningAgent
from rl_agent.strategy import EpsilonGreedyStrategy
from rl_agent.q_network import QNetwork
from rl_agent.state_encoder import encode_state, get_state_size


WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 1000
HUD_HEIGHT = 160
BOARD_AREA_WIDTH = WINDOW_WIDTH
BOARD_AREA_HEIGHT = WINDOW_HEIGHT - HUD_HEIGHT

MARGIN = 2
REPLAY_DELAY = 250  # milliseconds between replay moves

AGENT_DELAY = 250

BG_COLOR = (25, 25, 25)
WALL_COLOR = (60, 60, 60)
UNPAINTED_COLOR = (220, 220, 220)
PAINTED_COLOR = (100, 180, 255)
PLAYER_COLOR = (255, 120, 120)
GRID_LINE_COLOR = (35, 35, 35)
TEXT_COLOR = (240, 240, 240)
ACCENT_COLOR = (255, 220, 120)


def compute_cell_size(rows: int, cols: int) -> int:
    cell_w = BOARD_AREA_WIDTH // cols
    cell_h = BOARD_AREA_HEIGHT // rows
    return max(20, min(cell_w, cell_h))


def painted_count(env: AmazeEnv, state) -> int:
    return state.painted_mask.bit_count()


def total_paintable(env: AmazeEnv) -> int:
    return env.goal_mask.bit_count()


def draw_board(
    screen: pygame.Surface,
    env: AmazeEnv,
    state,
    cell_size: int,
    board_offset_x: int,
    board_offset_y: int,
) -> None:
    screen.fill(BG_COLOR)

    for r in range(env.rows):
        for c in range(env.cols):
            x = board_offset_x + c * cell_size
            y = board_offset_y + r * cell_size
            rect = pygame.Rect(x, y, cell_size, cell_size)

            if env.grid[r][c] == 0:
                color = WALL_COLOR
            else:
                bit = env.cell_to_bit[(r, c)]
                painted = (state.painted_mask >> bit) & 1
                color = PAINTED_COLOR if painted else UNPAINTED_COLOR

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_LINE_COLOR, rect, MARGIN)

    pr, pc = state.player
    center_x = board_offset_x + pc * cell_size + cell_size // 2
    center_y = board_offset_y + pr * cell_size + cell_size // 2
    radius = max(6, cell_size // 4)
    pygame.draw.circle(screen, PLAYER_COLOR, (center_x, center_y), radius)


def draw_hud(
    screen: pygame.Surface,
    env: AmazeEnv,
    state,
    font: pygame.font.Font,
    move_count: int,
    replaying: bool,
    agent_playing: bool,
    current_solution,
    difficulty_name: str,
) -> None:
    panel_y = BOARD_AREA_HEIGHT
    pygame.draw.rect(screen, BG_COLOR, (0, panel_y, WINDOW_WIDTH, HUD_HEIGHT))

    solved = env.is_goal(state)
    painted = painted_count(env, state)
    total = total_paintable(env)

    line1 = (
        f"Moves: {move_count}   Painted: {painted}/{total}   "
        f"Size: {env.rows}x{env.cols}   Difficulty: {difficulty_name}"
    )

    if solved:
        line2 = "Solved!"
        line3 = "R reset | N random solvable | 1 easy | 2 medium | 3 hard"
        color2 = ACCENT_COLOR
        color3 = TEXT_COLOR
    elif replaying:
        line2 = "AI is playing..."
        line3 = "R reset | N random solvable | 1 easy | 2 medium | 3 hard | ESC quit"
        color2 = TEXT_COLOR
        color3 = TEXT_COLOR
    elif agent_playing:
        line2 = "RL agent is playing..."
        line3 = "R reset | K Stop agent | ESC quit"
        color2 = TEXT_COLOR
        color3 = TEXT_COLOR
    elif current_solution is not None:
        line2 = f"BFS ready: {len(current_solution)} moves"
        line3 = (
            "Arrows/WASD move | R reset | N random solvable | "
            "1 easy | 2 medium | 3 hard | SPACE replay | I AI autoplay | L RL agent | ESC quit"
        )
        color2 = TEXT_COLOR
        color3 = TEXT_COLOR
    else:
        line2 = "Arrows/WASD move | R reset | N random solvable | 1 easy | 2 medium | 3 hard"
        line3 = "B solve | SPACE replay | I AI autoplay | L RL agent | ESC quit"
        color2 = TEXT_COLOR
        color3 = TEXT_COLOR

    surf1 = font.render(line1, True, TEXT_COLOR)
    surf2 = font.render(line2, True, color2)
    surf3 = font.render(line3, True, color3)

    screen.blit(surf1, (10, panel_y + 10))
    screen.blit(surf2, (10, panel_y + 45))
    screen.blit(surf3, (10, panel_y + 80))


def get_agent_action(agent, env, state) -> int:
    encoded_state = encode_state(env, state)
    return agent.select_action_from_state(encoded_state)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Amaze AI - Pygame Debug Viewer")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    key_to_action = {
        pygame.K_UP: UP,
        pygame.K_w: UP,
        pygame.K_DOWN: DOWN,
        pygame.K_s: DOWN,
        pygame.K_LEFT: LEFT,
        pygame.K_a: LEFT,
        pygame.K_RIGHT: RIGHT,
        pygame.K_d: RIGHT,
    }

    # Initial level
    grid, start = make_random_solvable_level(rows=7, cols=7, wall_prob=0.22)
    env = AmazeEnv(grid, start)
    state = env.reset()
    current_solution = None
    difficulty_name = "Random Solvable"

    cell_size = compute_cell_size(env.rows, env.cols)
    board_pixel_width = env.cols * cell_size
    board_pixel_height = env.rows * cell_size
    board_offset_x = (BOARD_AREA_WIDTH - board_pixel_width) // 2
    board_offset_y = (BOARD_AREA_HEIGHT - board_pixel_height) // 2

    move_count = 0
    replay_solution = []
    replay_index = 0
    replaying = False
    last_replay_time = 0

    input_size = get_state_size(env.rows, env.cols)
    q_network = QNetwork(input_size=input_size)
    agent = QLearningAgent(q_network=q_network, strategy=EpsilonGreedyStrategy(epsilon=0.5))

    agent_playing = False
    last_agent_time = 0
    agent_steps_taken = 0
    agent_step_limit = 100

    def load_level(new_grid, new_start, new_solution=None, new_difficulty="Random Solvable"):
        nonlocal env, state, cell_size, board_offset_x, board_offset_y
        nonlocal move_count, current_solution, replay_solution, replay_index, replaying
        nonlocal difficulty_name, agent, agent_playing, last_agent_time, agent_steps_taken

        env = AmazeEnv(new_grid, new_start)
        state = env.reset()

        input_size = get_state_size(env.rows, env.cols)
        q_network = QNetwork(input_size=input_size)
        agent = QLearningAgent(q_network=q_network, strategy=EpsilonGreedyStrategy(epsilon=0.5))

        cell_size = compute_cell_size(env.rows, env.cols)
        board_pixel_width = env.cols * cell_size
        board_pixel_height = env.rows * cell_size
        board_offset_x = (BOARD_AREA_WIDTH - board_pixel_width) // 2
        board_offset_y = (BOARD_AREA_HEIGHT - board_pixel_height) // 2

        move_count = 0
        current_solution = new_solution
        replay_solution = []
        replay_index = 0
        replaying = False
        difficulty_name = new_difficulty

        agent_playing = False
        last_agent_time = 0
        agent_steps_taken = 0
        agent_step_limit = 100

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    state = env.reset()
                    move_count = 0
                    current_solution = None
                    replay_solution = []
                    replay_index = 0
                    replaying = False
                    print("Reset level.")

                elif event.key == pygame.K_b:
                    current_solution = solve_bfs(env)
                    replaying = False
                    replay_solution = []
                    replay_index = 0

                    if current_solution is not None:
                        print("BFS solution:", [ACTION_NAMES[a] for a in current_solution])
                    else:
                        print("No BFS solution found.")

                elif event.key == pygame.K_SPACE:
                    if current_solution is not None:
                        state = env.reset()
                        move_count = 0
                        replay_solution = list(current_solution)
                        replay_index = 0
                        replaying = True
                        last_replay_time = now
                        print("Replaying saved BFS solution...")

                elif event.key == pygame.K_i:
                    current_solution = solve_bfs(env)

                    if current_solution is not None:
                        print("AI takeover solution:", [ACTION_NAMES[a] for a in current_solution])
                        state = env.reset()
                        move_count = 0
                        replay_solution = list(current_solution)
                        replay_index = 0
                        replaying = True
                        last_replay_time = now
                    else:
                        print("AI takeover failed: no solution found.")

                elif event.key == pygame.K_l:
                    agent_playing = True
                    replaying = False
                    replay_solution = []
                    replay_index = 0
                    current_solution = None
                    agent_steps_taken = 0
                    last_agent_time = now
                    print("RL agent playback started.")

                elif event.key == pygame.K_k:
                    agent_playing = False
                    print("RL agent playback stopped")

                elif event.key == pygame.K_n:
                    try:
                        grid, start = make_random_solvable_level(rows=7, cols=7, wall_prob=0.22)
                        load_level(grid, start, None, "Random Solvable")
                        print("Generated new random solvable level.")
                    except RuntimeError as e:
                        print(f"Level generation failed: {e}")

                elif event.key == pygame.K_1:
                    try:
                        grid, start, solution = make_easy_level()
                        load_level(grid, start, solution, "Easy")
                        print(f"Generated EASY level with {len(solution)} moves.")
                    except RuntimeError as e:
                        print(f"Easy generation failed: {e}")
                    except Exception as e:
                        print(f"Unexpected easy-level error: {e}")

                elif event.key == pygame.K_2:
                    try:
                        grid, start, solution = make_medium_level()
                        load_level(grid, start, solution, "Medium")
                        print(f"Generated MEDIUM level with {len(solution)} moves.")
                    except RuntimeError as e:
                        print(f"Medium generation failed: {e}")
                    except Exception as e:
                        print(f"Unexpected medium-level error: {e}")

                elif event.key == pygame.K_3:
                    try:
                        grid, start, solution = make_hard_level()
                        load_level(grid, start, solution, "Hard")
                        print(f"Generated HARD level with {len(solution)} moves.")
                    except RuntimeError as e:
                        print(f"Hard generation failed: {e}")
                    except Exception as e:
                        print(f"Unexpected hard-level error: {e}")

                elif event.key in key_to_action and not env.is_goal(state) and not replaying and not agent_playing:
                    action = key_to_action[event.key]
                    next_state = env.step(state, action).state

                    if next_state != state:
                        print(f"Manual move: {ACTION_NAMES[action]}")
                        state = next_state
                        move_count += 1

        if replaying and replay_index < len(replay_solution):
            if now - last_replay_time >= REPLAY_DELAY:
                action = replay_solution[replay_index]
                state = env.step(state, action).state
                move_count += 1
                replay_index += 1
                last_replay_time = now

        if replaying and replay_index >= len(replay_solution):
            replaying = False

        if agent_playing and not env.is_goal(state):
            if now - last_agent_time >= AGENT_DELAY:
                action = get_agent_action(agent, env, state)
                next_state = env.step(state, action).state

                if next_state != state:
                    state = next_state
                    move_count += 1
                
                agent_steps_taken += 1
                last_agent_time = now

                if agent_steps_taken >= agent_step_limit:
                    agent_playing = False
                    print("RL agent stopped: step limit reached.")

        if agent_playing and env.is_goal(state):
            agent_playing = False
            print("RL agent solved the level.")

        draw_board(screen, env, state, cell_size, board_offset_x, board_offset_y)
        draw_hud(screen, env, state, font, move_count, replaying, agent_playing, current_solution, difficulty_name)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()