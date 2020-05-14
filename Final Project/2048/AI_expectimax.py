import random
from inspect import signature
from typing import Callable, Union

import numpy as np
import pygame
from game import Game2048

_MOVES = ["Up", "Down", "Left", "Right"]

_KEYMAP = {"Up": pygame.K_UP, "Down": pygame.K_DOWN, "Left": pygame.K_LEFT, "Right": pygame.K_RIGHT}
_REVERSE_KEYMAP = {v: k for k, v in _KEYMAP.items()}

HEURISTICS = ["greedy", "safe", "safest", "monotonic", "smooth"]

class _Node(object):
    def __init__(self, parent=None):
        self.depth = 0
        self.num_visits = 0
        self.visit_score = 0
        self.avg_score = 0
        self.parent = parent
        super(_Node, self).__init__()

class StateNode(_Node):
    def __init__(self, state: np.ndarray, parent=None):
        self.state = state
        self.moves = {}  # Maps hashes of moves ("Up", "Down",...) to MoveNodes
        # self.unvisited = valid_moves(state)
        super(StateNode, self).__init__(parent=parent)

    def __hash__(self):
        return str(self.state.tolist()).__hash__()

    def add_move(self, move):
        if hash(move) not in self.moves:
            new_move = MoveNode(move, self)
            new_move.depth = self.depth
            self.moves.update({new_move.__hash__(): new_move})

    def select_next_move(self, max_score):
        moves = []
        UCT = []
        for _, v in self.moves.items():
            moves.append(v.move)
            score = v.avg_score / max_score + (2 * np.log(self.num_visits + 1) / self.num_visits) ** 0.5
            UCT.append(score)

        UCT = np.array(UCT)
        return moves[np.random.choice(np.flatnonzero(UCT == UCT.max()))]

    def get_best_move(self):
        moves = []
        scores = []
        for _, v in self.moves.items():
            moves.append(v.move)
            scores.append(v.avg_score)

        scores = np.array(scores)
        return moves[np.random.choice(np.flatnonzero(scores == scores.max()))]

class MoveNode(_Node):
    def __init__(self, move: str, parent=None):
        self.move = move
        self.states = {}  # Maps StateNode hashes to StateNodes
        super(MoveNode, self).__init__(parent=parent)

    def __hash__(self):
        return self.move.__hash__()

    def add_state(self, state: np.ndarray):
        if hash(str(state.tolist())) not in self.states:
            new_state = StateNode(state, self)
            new_state.depth = self.depth + 1
            self.states.update({new_state.__hash__(): new_state})

class expectimax(_Node):

    def __init__(self, grid: np.ndarray,depth, max_search_depth=20):
        self.root = StateNode(np.copy(grid))
        self.cur_node = self.root
        self.max_search_depth = max_search_depth
        self.last_move = None
        # True means the next agent is MAX
        self.player = True
        self.depth = depth
        self.directions = ['Left','Right','Up','Down']


    def Expectimax(self):
        if self.isend(self.cur_node.state):
            return self.heuristic(self.cur_node.state)
        elif self.player:
            self.player = False
            return self.max_value(self.cur_node.state)
        else:
            return self.exp_value(self.cur_node.state)


    def max_value(self,state):
        v = -np.inf
        succ = self.successor_max(state)
        for state in succ:
            v = max(self.heuristic(state),v)
        return v

    def exp_value(self,state):
        v = 0
        succ_2, succ_4 = self.successor_exp(state)
        for succ in succ_2:
            v += self.heuristic(succ) * 0.9
        for succ in succ_4:
            v += self.heuristic(succ) * 0.1

        return v/(len(succ_2)+len(succ_4))

    def isend(self, grid):
        return np.any(grid == 2048) or not self.successor_max(grid) or not self.successor_exp(grid)

    def successor_max(self, grid, cur_score=None):
        # given a grid, generate successors
        # put direction and grid in a dictionary

        succ =[]
        for direction in self.directions:
            merged = grid.copy()
            for c in range(grid.shape[1]):
                merged[:, c] = quick_merge_row(grid[:, c], direction == direction, cur_score)
            if ~np.all(grid == merged):
                succ.append(merged)

        return succ

    def successor_exp(self,grid,cur_score=None):

        succ_2 = []
        succ_4 = []
        for row in range(grid.shape[0]):
            for column in range(grid.shape[1]):
                if grid[row,column]==0:
                    temp = grid.copy()
                    temp[row, column] = 2
                    succ_2.append(temp)

                    temp1 = grid.copy()
                    temp1[row, column] = 4
                    succ_4.append(temp1)
        return succ_2,succ_4

    def heuristic(self,grid: np.ndarray):
        # calculated the empty space + heavy weights for largest values on the edge
        # number of possible merge

        build_list = [4 ** i for i in range(7)]
        weighted_matrix = np.array([build_list[i:i + 4] for i in range(4)]).reshape(4, 4)

        return np.sum(grid == 0) + np.sum(np.multiply(grid, weighted_matrix))


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


def quick_merge_row(row, right=True, old_score=None):
    """
    Quick merge for a single row, courtesy of
    https://stackoverflow.com/questions/22970210/most-efficient-way-to-shift-and-merge-the-elements-of-a-list-in-python-2048

    Works for columns as well (right = down in that case)

    :param row: Row of the game grid to merge
    :param right: Whether to merge to the right or to the left (right by default)
    :param old_score: The current game score if a new score should be calculated; None otherwise
    :return: The merged row if old_score is None; else a tuple of (merged_row, new_score)
    """
    if right:
        row = row[::-1]
    values = []
    empty = 0
    for n in row:
        if values and n == values[-1]:
            values[-1] = 2 * n
            if old_score is not None:
                old_score += 2 * n
            empty += 1
        elif n:
            values.append(n)
        else:
            empty += 1
    values += [0] * empty
    if right:
        values = values[::-1]
    return values if old_score is None else (values, old_score)

