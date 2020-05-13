import os
import sys
import argparse
import statistics as stats

import errno
import pygame
import numpy as np
from appdirs import user_data_dir

from game import Game2048
from manager import GameManager
import AI
import time


def run_game(game_class=Game2048, title='2048: In Python!', data_dir=None, **kwargs):
    pygame.init()
    pygame.display.set_caption(title)

    AI_type = kwargs["AI_type"]

    # Try to set the game icon.
    try:
        pygame.display.set_icon(game_class.icon(32))
    except pygame.error:
        # On windows, this can fail, so use GDI to draw then.
        print('Consider getting a newer card or drivers.')
        os.environ['SDL_VIDEODRIVER'] = 'windib'

    if data_dir is None:
        data_dir = user_data_dir(appauthor='Quantum', appname='2048', roaming=True)
        try:
            os.makedirs(data_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    score_file_prefix = os.path.join(data_dir, '2048')
    state_file_prefix = os.path.join(data_dir, '2048')

    if AI_type:
        score_file_prefix += '_' + AI_type
        state_file_prefix += '_' + AI_type

    if AI_type in ["heuristic", "rollout"]:
        if AI_type == "rollout" and kwargs["type"] is None or kwargs["type"] == 'None':
            type_str = "random"
        else:
            type_str = kwargs["type"]
        score_file_prefix += '_' + type_str
        state_file_prefix += '_' + type_str

    screen = pygame.display.set_mode((game_class.WIDTH, game_class.HEIGHT))
    manager = GameManager(Game2048, screen,
                          score_file_prefix + '.score',
                          state_file_prefix + '.%d.state', **kwargs)
    if not AI_type:
        try:
            while True:
                event = pygame.event.wait()
                manager.dispatch(event)
                for event in pygame.event.get():
                    manager.dispatch(event)
                manager.draw()

        finally:
            pygame.quit()
            manager.close()

    else:
        try:
            pygame.event.set_blocked([pygame.KEYDOWN, pygame.MOUSEBUTTONUP])
            manager.new_game(**kwargs)
            game_scores = []
            best_tiles = []
            condition = True
            tree = None

            if AI_type in ["rollout", "MCTS"]:
                num_rollouts = kwargs["num_rollouts"]
                max_depth = kwargs["max_depth"]
                epsilon = kwargs["epsilon"]
                if epsilon < 0 or epsilon > 1:
                    raise ValueError("Epsilon must be in the interval [0, 1].")
                if AI_type == "MCTS":
                    UCT = kwargs["UCT"]
                    tree = AI.GameTree(np.array(manager.game.grid), max_search_depth=max_depth,
                                       num_rollouts=num_rollouts, epsilon=epsilon, UCT=UCT)

            while condition:
                if manager.game.lost:
                    event = pygame.event.Event(pygame.MOUSEBUTTONUP, {"pos": manager.game.lost_try_again_pos})
                    game_scores.append(manager.game.score)
                    best_tiles.append(np.max(manager.game.grid))
                    if AI_type in ["random", "heuristic", "MCTS", "rollout"]:
                        condition = kwargs["num_games"] > len(game_scores)
                elif manager.game.won == 1:
                    event = pygame.event.Event(pygame.MOUSEBUTTONUP, {"pos": manager.game.keep_going_pos})
                elif AI_type == "random":
                    event = AI.random_move_event(np.array(manager.game.grid))
                elif AI_type == "heuristic":
                    event = AI.heuristic_move_event(np.array(manager.game.grid), kwargs["type"])
                elif AI_type == "rollout":
                    event = AI.rollouts(np.array(manager.game.grid), manager.game.score, kwargs["type"],
                                        max_search_depth=max_depth, num_rollouts=num_rollouts, epsilon=epsilon)
                elif AI_type == "MCTS":
                    event = tree.MCTS(np.array(manager.game.grid), manager.game.score)
                else:
                    raise ValueError("AI mode selected but invalid AI type was supplied!")
                manager.dispatch(event)
                manager.draw()

            print("Number of games played:", len(game_scores))
            print("Game Scores:")
            print(game_scores)
            print("Best Tiles:")
            print(best_tiles)
            print("Max Score:", max(game_scores))
            print("Max Tile:", max(best_tiles))
            print("Average Score:", stats.mean(game_scores))

        finally:
            if "simulate" not in kwargs:
                pygame.quit()
                manager.close()

        if "simulate" in kwargs:
            results = {
                "game_scores": game_scores,
                "best_tiles": best_tiles,
                "max_score": max(game_scores),
                "max_tile": max(best_tiles),
                "avg_score": stats.mean(game_scores)
            }
            return results


def main():
    # Parse command line args
    parser = argparse.ArgumentParser(description="Play 2048, or choose an AI to play instead!")
    parser.add_argument('--AI_type', action='store_true')
    subparsers = parser.add_subparsers(dest='AI_type')

    random_parser = subparsers.add_parser("random")
    random_parser.add_argument("num_games", nargs='?', default=10, type=int)

    heuristic_parser = subparsers.add_parser("heuristic")
    heuristic_parser.add_argument('-t', "--type", nargs='?', choices=["greedy", "safe", "safest", "monotonic",
                                                                      "smooth", "corner_dist"],
                                  default="safe", type=str)
    heuristic_parser.add_argument("num_games", nargs='?', default=10, type=int)

    MCTS_parser = subparsers.add_parser("MCTS")
    MCTS_parser.add_argument('-r', "--num_rollouts", nargs='?', default=100, type=int)
    MCTS_parser.add_argument('-d', "--max_depth", nargs='?', default=4, type=int)
    MCTS_parser.add_argument('-e', "--epsilon", nargs='?', default=0.1, type=float)
    MCTS_parser.add_argument('-U', "--UCT", action='store_true')
    MCTS_parser.add_argument("num_games", nargs='?', default=10, type=int)

    rollout_parser = subparsers.add_parser("rollout")
    rollout_parser.add_argument('-r', "--num_rollouts", nargs='?', default=500, type=int)
    rollout_parser.add_argument('-d', "--max_depth", nargs='?', default=4, type=int)
    rollout_parser.add_argument('-e', "--epsilon", nargs='?', default=0.1, type=float)
    rollout_parser.add_argument('-t', "--type", nargs='?', choices=["greedy", "safe", "safest", "monotonic",
                                                                    "smooth", "corner_dist"], default=None, type=str)
    rollout_parser.add_argument("num_games", nargs='?', default=10, type=int)

    kwargs = vars(parser.parse_args(sys.argv[1:]))
    
    start_time = time.time()
    run_game(**kwargs)
    end_time = time.time()
    print("total seconds for the simulation:", int(end_time-start_time))

