import random
from inspect import signature
from typing import Callable, Union

import numpy as np
import pygame
from game import Game2048

_MOVES = ["Up", "Down", "Left", "Right"]

_KEYMAP = {"Up": pygame.K_UP, "Down": pygame.K_DOWN, "Left": pygame.K_LEFT, "Right": pygame.K_RIGHT}


class _Node(object):
    def __init__(self):
        self.depth = 0
        self.num_visits = 0
        self.avg_score = 0
        self.parent = None
        super(_Node, self).__init__()


class StateNode(_Node):
    def __init__(self, state: np.ndarray):
        self.state = state
        self.moves = {}  # Maps strings ("Up", "Down",...) to MoveNodes
        super(StateNode, self).__init__()

    def __hash__(self):
        return str(self.state.tolist())


class MoveNode(_Node):
    def __init__(self, move: str):
        self.move = move
        self.states = {}  # Maps string representations of states to StateNodes
        super(MoveNode, self).__init__()

    def __hash__(self):
        return self.move


class GameTree(object):
    def __init__(self, grid: np.ndarray, max_search_depth=20, num_rollouts=500, epsilon=0.1):
        self.root = StateNode(np.copy(grid))
        self.cur_node = self.root
        super(GameTree, self).__init__()

        # TODO: Add function for MCTS, e.g. MCTS() -> move


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


def _heuristic_choose_direction(moves: list, heuristic_type="greedy"):
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
    elif heuristic_type == "greedy":
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


def random_move_event(game: Game2048):
    return pygame.event.Event(pygame.KEYDOWN, {"key": random.choice([_KEYMAP[move] for move in valid_moves(
        game.grid)])})


def quick_merge_row(row, right=True):
    """
    Quick merge for a single row, courtesy of
    https://stackoverflow.com/questions/22970210/most-efficient-way-to-shift-and-merge-the-elements-of-a-list-in-python-2048

    Works for columns as well (right = down in that case)

    :param row: Row of the game grid to merge
    :param right: Whether to merge to the right or to the left (right by default)
    :return: The merged row
    """
    if right:
        row = row[::-1]
    values = []
    empty = 0
    for n in row:
        if values and n == values[-1]:
            values[-1] = 2 * n
            empty += 1
        elif n:
            values.append(n)
        else:
            empty += 1
    values += [0] * empty
    if right:
        values = values[::-1]
    return values


def quick_merge(grid: np.ndarray, direction: str):
    merged = grid.copy()
    if direction in ["Up", "Down"]:
        for c in range(grid.shape[1]):
            merged[:, c] = quick_merge_row(grid[:, c], direction == "Down")

    else:
        for r in range(grid.shape[0]):
            merged[r, :] = quick_merge_row(grid[r, :], direction == "Right")

    return merged


def simulate_move(grid: np.ndarray, direction: str):
    grid = quick_merge(grid, direction)
    r, c = np.where(grid == 0)
    i = random.choice(range(len(r)))
    grid[r[i], c[i]] = 2 if random.random() < 0.9 else 4
    return grid


def is_valid_move(grid: np.ndarray, direction: str):
    test = simulate_move(grid, direction)
    return ~np.all(grid == quick_merge(grid, direction))


def valid_moves(grid: np.ndarray):
    return [move for move in _MOVES if is_valid_move(grid, move)]


def is_safe_move(grid: np.ndarray, direction: str):
    max_pos = np.unravel_index(grid.argmax(), grid.shape)
    new_max_pos = np.unravel_index(quick_merge(grid, direction).argmax(), grid.shape)
    return is_valid_move(grid, direction) and np.all(new_max_pos >= max_pos)


def safe_moves(grid: np.ndarray):
    return [move for move in _MOVES if is_safe_move(grid, move)]


def choose_min_move(grid: np.ndarray, moves: list, eval_func: Union[Callable[[np.ndarray], Union[int, float, complex]],
                                                                    Callable[[np.ndarray, np.ndarray],
                                                                             Union[int, float, complex]]]):
    """
    Choose a move to take based on an evaluation function. The move chosen will be the argmin of the function.
    :param grid: The current game grid
    :param moves: A list of moves to evaluate. Possible values are "Up", "Down", "Left", "Right"
    :param eval_func: The evaluation function, which will be evaluated using the current grid and/or the grids
                      resulting from making the moves in 'moves'. This can either have the signature eval_func(
                      new_grid) -> Number or eval_func(cur_grid, new_grid) -> Number.
    :return: The chosen move. This will be the argmin of 'eval_func' when it is evaluated for each move.
    """
    move_evals = []
    for move in moves:
        new_grid = quick_merge(grid, move)
        if len(signature(eval_func).parameters) == 2:
            move_evals.append(eval_func(grid, new_grid))
        else:
            move_evals.append(eval_func(new_grid))
    move_evals = np.array(move_evals)
    return moves[np.random.choice(np.flatnonzero(move_evals == move_evals.min()))]


def move_diff(cur_grid: np.ndarray, new_grid: np.ndarray):
    return int(np.sum(new_grid[new_grid != cur_grid]))


def smoothness(grid: np.ndarray):
    return np.abs(grid - np.pad(grid, ((1, 0), (0, 0)))[:-1, 0]) + \
           np.abs(grid - np.pad(grid, ((0, 1), (0, 0)))[1:, 0]) + \
           np.abs(grid - np.pad(grid, ((0, 0), (1, 0)))[0, :-1]) + \
           np.abs(grid - np.pad(grid, ((0, 0), (0, 1)))[0, 1:])


def monotonicity(grid: np.ndarray):
    return np.sum((grid < np.roll(grid, 1, axis=0))[1:, :]) + np.sum((grid < np.roll(grid, 1, axis=1))[:, 1:])


def heuristic_move_event(game: Game2048, heuristic_type="greedy"):
    grid = np.array(game.grid)
    if heuristic_type in ["greedy", "safe", "safest"]:
        moves = [_heuristic_choose_direction(move, heuristic_type) for move in _get_merge_directions(grid)]
        moves = np.array(moves)
        inds = grid.argsort(axis=None)[::-1]
        cell_move_priority = inds[grid.flatten()[inds] != 0]
        for move_ind in cell_move_priority:
            if moves[move_ind] == "Up":
                if heuristic_type == "greedy":
                    return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP})
                else:
                    # If up is an option, there is a companion tile that can merge down
                    return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN})
            elif moves[move_ind] == "Down":
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN})
            elif moves[move_ind] == "Left":
                if heuristic_type == "greedy":
                    return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT})
                else:
                    # If left is an option, there is a companion tile that can merge right
                    return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT})
            elif moves[move_ind] == "Right":
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT})

        if heuristic_type == "safe":
            valid = valid_moves(grid)
            safe = safe_moves(grid)
            if safe:
                return pygame.event.Event(pygame.KEYDOWN, {"key": random.choice([_KEYMAP[move] for move in safe])})
            else:
                return pygame.event.Event(pygame.KEYDOWN, {"key": random.choice([_KEYMAP[move] for move in valid])})

        elif heuristic_type == "safest":
            valid = valid_moves(grid)
            safe = safe_moves(grid)
            if safe:
                return pygame.event.Event(pygame.KEYDOWN, {"key": _KEYMAP[choose_min_move(grid, safe, move_diff)]})
            else:
                return pygame.event.Event(pygame.KEYDOWN, {"key": _KEYMAP[choose_min_move(grid, valid, move_diff)]})

        else:
            return pygame.event.Event(pygame.KEYDOWN, {"key": random.choice([_KEYMAP[move] for move in valid_moves(grid)])})

    elif heuristic_type == "monotonic":
        valid = valid_moves(grid)
        return pygame.event.Event(pygame.KEYDOWN, {"key": _KEYMAP[choose_min_move(grid, valid, monotonicity)]})

    else:  # Smooth
        valid = valid_moves(grid)
        return pygame.event.Event(pygame.KEYDOWN, {"key": _KEYMAP[choose_min_move(grid, valid, smoothness)]})
