import csv
import numpy as np
import copy


def printBoard(boardState):
    print(np.sqrt(boardState.state).astype(int))
    return True


def printMoves(moves):
    for move in moves:
        print(np.sqrt(move).astype(int))
        print("")
    return True


def readBoard(fileName):
    validList = [str(i) for i in range(10)]
    boardState = []

    with open(fileName, "r") as f:
        csvReader = csv.reader(f)
        for row in csvReader:
            boardState.append([int(item) ** 2 if item in validList else 0 for item in row])

    returnBoard = board(np.array(boardState))
    return returnBoard


class board(object):
    """docstring for board"""

    def __init__(self, state=None):
        """
        :type state: np.array
        """
        super(board, self).__init__()
        # state is a 2d ndarray
        # 0 means empty
        # non zeros indicate weights^2 of queens
        self.state = state
        self.priority = 0
        self.prev_state = copy.copy(state)
        if state is not None:
            lighter_attack_list = self.attackPieceList(min)
            self._h1_stored = min(lighter_attack_list) if len(lighter_attack_list) > 0 else 0
            self._h2_stored = sum(lighter_attack_list) if len(lighter_attack_list) > 0 else 0
            heavier_attack_list = self.attackPieceList(max)
            self._h3_stored = min(heavier_attack_list) if len(heavier_attack_list) > 0 else 0
            self._height = state.shape[0]
            self._width = state.shape[1]  # this is also #Queens
        else:
            self._width = None
            self._height = None

    # position of all queens
    @property
    def queenPos(self):
        return np.argwhere(self.state > 0)

    # three heuristic values
    @property
    def h1(self):
        if np.array_equal(self.state, self.prev_state):
            return self._h1_stored
        else:
            lighter_attack_list = self.attackPieceList(min)
            if self.prev_state is None:
                self._h2_stored = sum(lighter_attack_list) if len(lighter_attack_list) > 0 else 0
                heavier_attack_list = self.attackPieceList(max)
                self._h3_stored = min(heavier_attack_list) if len(heavier_attack_list) > 0 else 0
            self._h1_stored = min(lighter_attack_list) if len(lighter_attack_list) > 0 else 0
            self.prev_state = copy.copy(self.state)
            return self._h1_stored

    @property
    def h2(self):
        if np.array_equal(self.state, self.prev_state):
            return self._h2_stored
        else:
            lighter_attack_list = self.attackPieceList(min)
            if self.prev_state is None:
                self._h1_stored = min(lighter_attack_list) if len(lighter_attack_list) > 0 else 0
                heavier_attack_list = self.attackPieceList(max)
                self._h3_stored = min(heavier_attack_list) if len(heavier_attack_list) > 0 else 0
            self._h2_stored = sum(lighter_attack_list) if len(lighter_attack_list) > 0 else 0
            self.prev_state = copy.copy(self.state)
            return self._h2_stored

    @property
    def h3(self):
        if np.array_equal(self.state, self.prev_state):
            return self._h3_stored
        else:
            heavier_attack_list = self.attackPieceList(max)
            if self.prev_state is None:
                lighter_attack_list = self.attackPieceList(min)
                self._h1_stored = min(lighter_attack_list) if len(lighter_attack_list) > 0 else 0
                self._h2_stored = sum(lighter_attack_list) if len(lighter_attack_list) > 0 else 0
            self._h3_stored = min(heavier_attack_list) if len(heavier_attack_list) > 0 else 0
            self.prev_state = copy.copy(self.state)
            return self._h3_stored

    @property
    def height(self):
        if self._height is None and self.state is not None:
            self._height = self.state.shape[0]
        return self._height

    @property
    def width(self):
        if self._width is None and self.state is not None:
            self._width = self.state.shape[1]
        return self._width

    def ifAttack(self, index1, index2):
        # check if two pieces are attacking each other
        # index: np.ndarray [row,col] of queens

        # this method does not check for same queen (invalid input)
        # this method does not check if there is/are a/two queen(s)

        row1, col1 = index1
        row2, col2 = index2

        # horizontal
        if row1 == row2:
            return True
        # diagonal
        if abs(row1 - row2) == abs(col1 - col2):
            return True
        return False

    def ifValidQueen(self, row, col):
        # check if the given position has a queen
        return True if self.state[row, col] > 0 else False

    def attackPieceList(self, compare_func):
        # returns the list of lighter weight^2 of all attacking pairs
        returnList = []
        for index1 in range(self.queenPos.shape[0]):
            for index2 in range(index1 + 1, self.queenPos.shape[0]):
                pos1 = self.queenPos[index1]
                pos2 = self.queenPos[index2]
                if self.ifAttack(pos1, pos2):
                    weight1 = self.state[pos1[0], pos1[1]]
                    weight2 = self.state[pos2[0], pos2[1]]
                    returnList.append(compare_func(weight1, weight2))
        return returnList

    def cost(self, row, col, move):
        # row: row index of queen
        # col: col index of queen
        # move: distance to move

        # this method does not check valid moves
        # this method does not check for a queen
        return self.state[row, col] * abs(move)

    def ifValidMove(self, start_index, move):
        # move: - up, + down
        # not moving is an invalid move
        if move == 0:
            return False
        # we can only move queens
        if not self.ifValidQueen(start_index[0], start_index[1]):
            return False
        # we cannot move out of boundary
        if (start_index[0] + move) < 0 or (start_index[0] + move) > self.height - 1:
            return False
        return True

    def heuristic(self, h_type):
        if h_type == "h1":
            return self.h1
        elif h_type == "h2":
            return self.h2
        else:
            return self.h3

    def get_neighbors(self):
        returnList = []
        for pos in self.queenPos:
            for move in range(-self.height + 1, self.height):
                if self.ifValidMove(pos, move):
                    row, col = pos
                    weight = self.state[row, col]
                    new_state = copy.deepcopy(self.state)

                    new_state[row, col] = 0
                    new_state[row + move, col] = weight

                    returnList.append((board(new_state), self.cost(row, col, move)))
        return returnList

    def make_move(self, col, pos):
        queen_pos = self.state[:, col].nonzero()[0][0]
        if not self.ifValidMove((queen_pos, col), pos - queen_pos):
            return False
        else:
            self.state[queen_pos, col], self.state[pos, col] = self.state[pos, col], self.state[queen_pos, col]
            return True

    def get_neighbors_in_place(self, h_type):
        # return a list of dicts. all we care about currently is the column, start, stop, cost, and heuristic values
        move_list = []
        for queen_pos in self.queenPos:
            valid_moves = set(range(self.width)) - {queen_pos[0]}
            temp_pos = queen_pos[0]
            for new_pos in valid_moves:
                move_dict = {"col": queen_pos[1], "start_pos": queen_pos[0], "end_pos": new_pos, "move_cost":
                             self.cost(temp_pos, queen_pos[1], new_pos - queen_pos[0])}
                self.make_move(queen_pos[1], new_pos)
                temp_pos = new_pos
                move_dict["function_value"] = self.heuristic(h_type)
                move_list.append(move_dict)
            self.make_move(queen_pos[1], queen_pos[0])
        return move_list

    def __lt__(self, other):
        return self.priority < other.priority
