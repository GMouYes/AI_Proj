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
        self.unvisited = valid_moves(state)
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

    def __init__(self, grid: np.ndarray, maxplayer,depth, max_search_depth=20):
        self.root = StateNode(np.copy(grid))
        self.cur_node = self.root
        self.max_search_depth = max_search_depth
        self.last_move = None
        self.player = maxplayer
        self.depth = depth


    def Expectimax(self):

        return

    def max_value(self):
        v = -np.inf

        return v

    def exp_value(self):
        v = 0

        return v

    def isend(self):

        return

    def successor(self):

        return


    def heuristic(grid: np.ndarray):
        # calculated the empty space + heavy weights for largest values on the edge
        # number of possible merge

        build_list = [4 ** i for i in range(7)]
        weighted_matrix = np.array([build_list[i:i + 4] for i in range(4)]).reshape(4, 4)

        return np.sum(grid == 0) + np.sum(np.multiply(grid, weighted_matrix))