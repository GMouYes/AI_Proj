import os
import sys
import argparse
import statistics as stats

import errno
import pygame
from appdirs import user_data_dir

from game import Game2048
from manager import GameManager
import AI


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

    if AI_type == "heuristic":
        score_file_prefix += str(kwargs["type"])
        state_file_prefix += str(kwargs["type"])

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
            game_scores = []
            condition = True

            while condition:
                if manager.game.lost:
                    event = pygame.event.Event(pygame.MOUSEBUTTONUP, {"pos": manager.game.lost_try_again_pos})
                    game_scores.append(manager.game.score)
                    if AI_type in ["random", "heuristic"]:
                        condition = kwargs["num_games"] > len(game_scores)
                elif manager.game.won == 1:
                    event = pygame.event.Event(pygame.MOUSEBUTTONUP, {"pos": manager.game.keep_going_pos})
                elif AI_type == "random":
                    event = AI.random_move_event(manager.game)
                elif AI_type == "heuristic":
                    event = AI.heuristic_move_event(manager.game, kwargs["type"])
                else:
                    raise ValueError("AI mode selected but invalid AI type was supplied!")
                manager.dispatch(event)
                manager.draw()

            pygame.quit()
            manager.close()
            print("Number of games played:", len(game_scores))
            print("Max Score:", max(game_scores))
            print("Average Score:", stats.mean(game_scores))

        finally:
            pygame.quit()
            manager.close()


def main():
    # Parse command line args
    parser = argparse.ArgumentParser(description="Play 2048, or choose an AI to play instead!")
    parser.add_argument('--AI_type', action='store_true')
    subparsers = parser.add_subparsers(dest='AI_type')

    random_parser = subparsers.add_parser("random")
    random_parser.add_argument("num_games", nargs='?', default=10)

    heuristic_parser = subparsers.add_parser("heuristic")
    heuristic_parser.add_argument('-t', "--type", nargs='?', choices=[1, 2], default=2, type=int)
    heuristic_parser.add_argument("num_games", nargs='?', default=10)
    kwargs = vars(parser.parse_args(sys.argv[1:]))

    run_game(**kwargs)
