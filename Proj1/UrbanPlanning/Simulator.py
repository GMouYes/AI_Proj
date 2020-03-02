import Rules
import algo
import numpy as np
import pandas as pd
import itertools
import copy


def gen_random_map(size, num_s, num_x, max_c, max_i, max_r):
    map_state = np.random.randint(1, 10, size=size).astype('<U1')
    site_dict = {"S": num_s, "X": num_x}
    for site_name in site_dict:
        for site in range(site_dict[site_name]):
            done = False
            new_pos = ()
            while not done:
                new_pos = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
                if map_state[new_pos[0], new_pos[1]] not in ["S", "X"]:
                    done = True
            map_state[new_pos[0], new_pos[1]] = site_name

    return Rules.Map(map_state, max_i, max_c, max_r)


def simulate_greedy_hill_climbing(sizes, num_s, num_x, max_c, max_i, max_r, var_vals, print_every_n_steps=10,
                                  replications=10):

    vars = var_vals.keys()
    vals = var_vals.values()
    result_table = pd.DataFrame(columns=(["size", "elapsedTime", "score"] + list(vars)))
    iteration = 0
    sizes = sorted(sizes * replications)
    cur_size = sizes[0]
    cur_num_s = num_s[0]
    cur_num_x = num_x[0]

    urbanmap = gen_random_map(cur_size, cur_num_s, cur_num_x, max_c[0], max_i[0], max_r[0])
    for size_ind in range(len(sizes)):
        if sizes[size_ind] != cur_size and cur_num_s != num_s[size_ind] and cur_num_x != num_x[size_ind]:
            urbanmap = gen_random_map(sizes[size_ind], num_s[size_ind], num_x[size_ind], max_c[size_ind],
                                      max_i[size_ind], max_r[size_ind])
        urbanmap.maxCommercial = max_c[size_ind]
        urbanmap.maxIndustrial = max_i[size_ind]
        urbanmap.maxResidential = max_r[size_ind]
        for val_combo in itertools.product(*vals):
            kw_dict = dict(zip(vars, val_combo))
            result = algo.greedyHillClimb(copy.copy(urbanmap), **kw_dict)
            result_dict = {"size": sizes[size_ind], "elapsedTime": result["elapsedTime"], "score": result["score"]}
            result_dict.update(kw_dict)

            if iteration % print_every_n_steps == 0:
                print(result_dict)
            result_table = pd.concat([result_table, pd.DataFrame.from_dict([result_dict])])
            iteration += 1
        cur_size = sizes[size_ind]
        cur_num_s = num_s[size_ind]
        cur_num_x = num_x[size_ind]
    return result_table


def run_greedy_hill_climbing_simulation():
    sizes = [(5, 6)]
    num_s = [4]
    num_x = [3]
    max_c = [1]
    max_i = [2]
    max_r = [2]
    param_dict = {
        "confidence_thresh": range(10, 30, 10),
        "max_sideways_moves": range(0, 30, 10),
        "initial_temp": range(10, 101, 10),
        "cooling_schedule": ["geom"],
        "cooling_param": np.arange(0.1, 1.1, 0.1)
    }
    return simulate_greedy_hill_climbing(sizes, num_s, num_x, max_c, max_i, max_r, param_dict, replications=1)


def main():
    run_greedy_hill_climbing_simulation()


if __name__ == '__main__':
    main()
