import numpy as np

_MOVES = ["Up", "Down", "Left", "Right"]


class Expectimax:

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def expectimax(self, current_depth, state: np.ndarray, is_max_turn):
        if current_depth == self.max_depth or is_end(state, is_max_turn):
            # return evaluation function(utility)
            return heuristic(state)

        # ai's turn
        if is_max_turn:
            # get possible next action
            moves = valid_moves(state)
            # next_states = []

            max_utility = float('-inf')
            best_move = None

            for move in moves:
                # next_state = np.copy(state)
                # make the move
                # next_state.move(move)
                next_state = quick_merge(state, move)
                child_utility = self.expectimax(current_depth + 1, next_state, not is_max_turn)
                if child_utility > max_utility:
                    max_utility = child_utility
                    best_move = move

            return max_utility, best_move


        # computer's turn, insert 2 or 4
        else:
            empty_cells = get_empty_cells(state)
            empty_num = len(empty_cells)
            chance_2, chance_4 = 0.9 / empty_num, 0.1 / empty_num

            tiles = []
            for empty_cell in empty_cells:
                tiles.append((empty_cell, 2, chance_2))
                tiles.append((empty_cell, 4, chance_4))

            chance_utility = 0

            for tile in tiles:
                # insert the tile
                next_state = insert_tile(state, tile[0], tile[1])
                utility, _ = self.expectimax(current_depth + 1, next_state, not is_max_turn)
                chance_utility += utility * tile[2]

            return chance_utility

    def get_best_move(self, state):
        return self.expectimax(0, state, True)


# get_all_empty_cells
def get_empty_cells(state: np.ndarray):
    cells = []
    for x in range(4):
        for y in range(4):
            if state[x][y] == 0:
                cells.append((x, y))
    return cells


# return a new state after insertion
def insert_tile(state: np.ndarray, position, value):
    new_state = state.copy()
    new_state[position[0]][position[1]] = value
    return new_state


def is_valid_move(grid: np.ndarray, direction: str):
    return ~np.all(grid == quick_merge(grid, direction))


def valid_moves(grid: np.ndarray):
    return [move for move in _MOVES if is_valid_move(grid, move)]


def quick_merge(grid: np.ndarray, direction: str, cur_score=None):
    merged = grid.copy()
    if direction in ["Up", "Down"]:
        for c in range(grid.shape[1]):
            if cur_score is None:
                merged[:, c] = quick_merge_row(grid[:, c], direction == "Down", cur_score)
            else:
                merged[:, c], cur_score = quick_merge_row(grid[:, c], direction == "Down", cur_score)

    else:
        for r in range(grid.shape[0]):
            if cur_score is None:
                merged[r, :] = quick_merge_row(grid[r, :], direction == "Right", cur_score)
            else:
                merged[r, :], cur_score = quick_merge_row(grid[r, :], direction == "Right", cur_score)

    return merged if cur_score is None else (merged, cur_score)


def quick_merge_row(row, right=True, old_score=None):
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


def is_end(state: np.ndarray, is_max_turn: bool):
    return len(valid_moves(state)) == 0 and is_max_turn


def heuristic(grid: np.ndarray):
    # calculated the empty space + heavy weights for largest values on the edge
    # number of possible merge

    build_list = [4 ** i for i in range(7)]
    weighted_matrix = np.array([build_list[i:i + 4] for i in range(4)]).reshape(4, 4)
    return np.sum(grid == 0) + np.sum(np.multiply(grid, weighted_matrix))
