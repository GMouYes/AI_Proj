class State:

    def __init__(self):







class Expectimax:

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def expectimax(self, current_depth, state, is_max_turn):
        if current_depth == self.max_depth or state.isend():
#             return evaluation function(utility)
            pass
        # ai's turn
        if is_max_turn:
            # get possible next action
            moves = get_possible_moves()
            # next_states = []

            max_utility = float('-inf')
            # move = None

            for move in moves:
                next_state = np.copy(state)
                # make the move
                next_state.move(move)
                child_utility, _ = self.expectimax(current_depth+1, next_state, not is_max_turn)
                if child_utility > max_utility:
                    max_utility = child_utility

            return max_utility

            # for next_state in next_states:
            #     child_utility, _ = self.expectimax(current_depth+1, next_state[1], not is_max_turn)
            #     if child_utility > max_utility:
            #         max_utility = child_utility
            #         move = next_state[0]
            #
            # return max_utility, move

        # computer's turn, insert 2 or 4
        else:
            empty_cells = get_empty_cells()
            empty_num = len(empty_cells)
            chance_2, chance_4 = 0.9/empty_num, 0.1/empty_num

            tiles = []
            for empty_cell in empty_cells:
                tiles.append((empty_cell, 2, chance_2))
                tiles.append((empty_cell, 4, chance_4))


            chance_utility = 0

            for tile in tiles:
                next_state = np.copy(state)
                # insert the tile, position and num
                next_state.insert_tile(tile[0], tile[1])
                utility, _ = self.expectimax(current_depth+1, next_state, not is_max_turn)
                chance_utility += utility*tile[2]

            return chance_utility








