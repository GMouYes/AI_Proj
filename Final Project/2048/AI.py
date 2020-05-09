import random
import numpy as np
import pygame
from game import Game2048


def _get_merge_directions(grid: np.ndarray):
    move_list = [[] for _ in range(grid.size)]
    ind_array = np.arange(grid.size).reshape(grid.shape)

    for r in range(grid.shape[0]):
        row = grid[r, :]
        inds = ind_array[r, :][row != 0]
        row = row[row != 0]
        if row.size >= 2:
            for i in range(row.size):
                value = row[i]
                if i > 0 and inds[i] % grid.shape[0] != 0 and row[i - 1] == value:  # Left
                    move_list[inds[i]].append("Left")
                if i < row.size - 1 and (inds[i] + 1) % grid.shape[0] != 0 and row[i + 1] == value:  # Right
                    move_list[inds[i]].append("Right")

    for c in range(grid.shape[1]):
        col = grid[:, c]
        inds = ind_array[:, c][col != 0]
        col = col[col != 0]
        if col.size >= 2:
            for i in range(col.size):
                value = col[i]
                if i > 0 and inds[i] % grid.shape[1] != 0 and col[i - 1] == value:  # Up
                    move_list[inds[i]].append("Up")
                if i < col.size - 1 and (inds[i] + 1) % grid.shape[1] != 0 and col[i + 1] == value:  # Down
                    move_list[inds[i]].append("Down")

    return move_list


def _heuristic_choose_direction(moves: list, heuristic_type=1):
    """
    Given a list of possible merge directions, chooses a direction to move. Heuristic type 1 picks any possible
    merge; heuristic type 2 prioritizes moving to the bottom right, since grouping the largest tiles is an effective
    strategy.

    :param moves: A list of possible merge directions from _get_merge_directions
    :param heuristic_type: The heuristic type to use. See the README for more info.
    :return: A direction, either "Up", "Down", "Left", or "Right", or False if no merges are possible.
    """
    if len(moves) == 0:
        return False
    elif heuristic_type == 1:
        return random.choice(moves)
    else:
        if "Down" in moves:
            if "Right" in moves:
                return random.choice(["Down", "Right"])
            return "Down"
        elif "Right" in moves:
            return "Right"
        else:
            return random.choice(moves)


def random_move_event():
    return pygame.event.Event(pygame.KEYDOWN, {"key": random.choice([pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT,
                                                                     pygame.K_RIGHT])})


def heuristic_move_event(game: Game2048, heuristic_type=1):
    grid = np.array(game.grid)
    moves = [_heuristic_choose_direction(move, heuristic_type) for move in _get_merge_directions(grid)]
    moves = np.array(moves)
    inds = grid.argsort(axis=None)[::-1]
    cell_move_priority = inds[grid.flatten()[inds] != 0]
    for move_ind in cell_move_priority:
        if moves[move_ind] == "Up":
            if heuristic_type == 1:
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP})
            else:
                # If up is an option, there is a companion tile that can merge down
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN})
        elif moves[move_ind] == "Down":
            return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN})
        elif moves[move_ind] == "Left":
            if heuristic_type == 1:
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT})
            else:
                # If left is an option, there is a companion tile that can merge right
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT})
        elif moves[move_ind] == "Right":
            return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT})

    if heuristic_type == 2:
        max_pos = np.unravel_index(grid.argmax(), grid.shape)
        if np.sum(grid[:, max_pos[1]] != 0) == grid.shape[1]:
            if np.sum(grid[max_pos[0], :] != 0) == grid.shape[0]:
                return pygame.event.Event(pygame.KEYDOWN, {"key": random.choice([pygame.K_UP, pygame.K_DOWN,
                                                                                 pygame.K_LEFT, pygame.K_RIGHT])})
            else:
                return pygame.event.Event(pygame.KEYDOWN, {"key": random.choice([pygame.K_UP, pygame.K_DOWN])})
        else:
            return pygame.event.Event(pygame.KEYDOWN, {"key": random.choice([pygame.K_LEFT, pygame.K_RIGHT])})

    else:
        return pygame.event.Event(pygame.KEYDOWN, {"key": random.choice([pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT,
                                                                     pygame.K_RIGHT])})
