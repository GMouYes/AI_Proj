import Board
import searchAlgo
import numpy as np
import pandas as pd
import itertools
import copy


def gen_random_board(size, skew=False):
    if size < 4:
        raise ValueError("Boards of size 3 or less have no solutions.")
    new_board = Board.board()
    done = False
    while not done:
        state = np.zeros((size, size), dtype='int32')
        positions = np.random.randint(0, size - 1, size=size, dtype='int32')
        if skew:
            weights = [1, 9, 2, 8, 3, 7, 4, 6, 5]
            weights = weights * (size // len(weights)) + weights[:size % len(weights)]
            np.random.shuffle(weights)

        else:
            weights = np.random.randint(1, 9, size=size)

        for i in range(len(positions)):
            state[positions[i], i] = weights[i]

        new_board.state = state
        if new_board.h1 > 0:
            done = True

    return new_board


def simulate_greedy_hill_climbing(sizes, h_type, var_vals, print_every_n_steps=10, replications=10):
    """

    :param sizes:
    :param h_type:
    :param var_vals:
    :type sizes: list(int)
    :type h_type: str
    :type var_vals: dict
    :return:
    """
    vars = var_vals.keys()
    vals = var_vals.values()
    result_table = pd.DataFrame(columns=(["size", "elapsedTime", "cost", "solved"] + list(vars)))
    iteration = 0
    sizes = sorted(sizes * replications)
    cur_size = sizes[0]
    board = gen_random_board(cur_size, True)
    for size in sizes:
        if size != cur_size:
            board = gen_random_board(size, True)
        for val_combo in itertools.product(*vals):
            kw_dict = dict(zip(vars, val_combo))
            result = searchAlgo.greedyHillClimb(copy.copy(board), h_type, **kw_dict)
            result_dict = {"size": size, "elapsedTime": result["elapsedTime"], "cost": result["cost"],
                           "solved": result["solved"]}
            result_dict.update(kw_dict)

            if iteration % print_every_n_steps == 0:
                print(result_dict)
            result_table = pd.concat([result_table, pd.DataFrame.from_dict([result_dict])])
            iteration += 1
        cur_size = size
    return result_table


def run_greedy_hill_climbing_simulation():
    sizes = [4]
    h_type = "h1"
    param_dict = {
        # "confidence_thresh": range(10, 101, 10),
        # "max_sideways_moves": range(0, 101, 10),
        "initial_temp": range(10, 101, 10),
        "cooling_schedule": ["geom"],
        "cooling_param": np.arange(0.1, 1.1, 0.1)
    }
    return simulate_greedy_hill_climbing(sizes, h_type, param_dict)


def main():
    run_greedy_hill_climbing_simulation()


if __name__ == '__main__':
    main()
