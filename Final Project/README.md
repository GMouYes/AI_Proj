# 2048 AI

Built largely from the source code of the [PyPI 2048 package](https://github.com/quantum5/2048).

![2048 Preview](https://guanzhong.ca/assets/projects/2048-2fd91615603e0f5fed0299df4524c4494968c7b1d762cbb0209354cfa2215639.png)

## Dependencies

In addition to a working Python installation (we used the latest Python 3.7 Anaconda distribution), the following
packages are required:
* `pygame`
* `appdirs`

Both are available in the PyPI and can be installed using `pip`.

## Usage

Currently, the script can be run as follows, with optional arguments in brackets:

`python __main__.py [-h|--help] [--AI_type] {random,heuristic, MCTS, rollout} ...`, where
* `-h|--help`: Displays command help
* `--AI_type`: If supplied, a valid AI type and the associated parameters must be supplied; else, the game starts
normally, with full human control. Valid types are:
    * `random`: Makes random moves. Possible arguments are `... random [-h|--help] [num_games]`:
        * `-h|--help`: Displays command help
        * `num_games`: The number of games for the AI to play. The default is 10.
    * `heuristic`: Uses a naive heuristic-based AI to play games. Possible arguments are
    `... heuristic [-h|--help] [-t|--type {greedy, safe, safest, monotonic, smooth, corner_dist, expert}] [num_games]`:
        * `-h|--help`: Displays command help
        * `-t|--type {greedy, safe, safest, monotonic, smooth, corner_dist}`: The type of heuristic to use.
            * `greedy` makes any merge it can.
            * `safe` tries to play a bit smarter, moving the highest tile in the bottom right and keeping it there,
            if possible.
            * `safest` takes caution a bit further. When no merges are possible and there are multiple "safe" moves to
            choose from, it picks the move that moves the group of tiles with the lowest value (moving larger tiles is
            more risky).
            * `monotonic` prioritizes monotonicity. Tiles should be increasing across rows and down columns.
            * `smooth` prioritizes smoothness. The agent will attempt to keep tiles of the same value adjacent to each
            other, since this can lead to merging opportunities.
            * `corner_dist` penalizes high-valued tiles far from the bottom-right corner. The agent tries to minimize
            the sum of (*tile_value* x *Manhattan_distance_from_bottom_right*).
            * `expert` is a scored hybrid of `monotonic` and `smooth`, prioritizing both monotonicity
            vertically/horizontally while also rewarding merge opportunities.
            
            The default is type `safe`.
        * `num_games`: The number of games for the AI to play. The default is 10.
        
    * `MCST`: Monte-Carlo Tree Search. Possible arguments are `... MCST [-h|--help] [-r|--num_rollouts [NUM_ROLLOUTS]]
    [-d|--max_depth [MAX_DEPTH]] [-e|--epsilon[EPSILON]] [-U|--UCT] [--use_expert] [num_games]`:
        * `-h|--help`: Displays command help
        * `-r|--num_rollouts [NUM_ROLLOUTS]`: The number of simulations to run per move. Default is 100.
        * `-d|--max_depth [MAX_DEPTH]`: The maximum number of moves to run per simulation. Default is 4.
        * `-e|--epsilon [EPSILON]`: The exploration rate; the chance of making a random move during a simulation instead
        of the known best. Default is 0.
        * `-U|--UCT`: Whether to use Upper Confidence bounds for Trees to choose whether to explore or exploit.
        Default is False.
        * `-t|--type {greedy, safe, safest, monotonic, smooth, corner_dist, expert}`: The heuristic to use during
        the simulation phase when choosing moves to explore. All options are the same as `heuristic` above.
        If no type is supplied, the agent chooses randomly.
        * `--use_expert`: If supplied, uses the heuristic score from the `expert` heuristic to score board states,
        instead of the actual game score. This can lead to more cautious behavior. The default is False.
        * `num_games`: The number of games for the AI to play. The default is 10.
        
    * `rollout`: Instead of building a game tree, use rollouts to predict how well possible moves will do, with
    preference potentially governed by a heuristic. Possible arguments are `... rollout [-h|--help] [-r|--num_rollouts [NUM_ROLLOUTS]]
    [-d|--max_depth [MAX_DEPTH]] [-e|--epsilon[EPSILON]]
    [-t|--type {greedy, safe, safest, monotonic, smooth, corner_dist, expert}] [--use_expert] [num_games]`:
        * `-h|--help`: Displays command help
        * `-r|--num_rollouts [NUM_ROLLOUTS]`: The number of simulations to run per move. Default is 500.
        * `-d|--max_depth [MAX_DEPTH]`: The maximum number of moves to run per simulation. Default is 4.
        * `-e|--epsilon [EPSILON]`: The exploration rate; the chance of making a random move during a simulation instead
        of the known best. Default is 0.
        * `-t|--type {greedy, safe, safest, monotonic, smooth, corner_dist, expert}`: The heuristic to use during
        rollouts when "exploiting" knowledge of the game. All options are the same as `heuristic` above.
        If no type is supplied, the agent chooses randomly.
        * `--use_expert`: If supplied, uses the heuristic score from the `expert` heuristic to score board states,
        instead of the actual game score. This can lead to more cautious behavior. The default is False.
        * `num_games`: The number of games for the AI to play. The default is 10.
        
    * `expectimax`: expectimax Search. Possible arguments are `... expectimax [-h|--help][-d|--max_depth [MAX_DEPTH]] ][num_games]`:
        * `-h|--help`: Displays command help
        * `-d|--max_depth [MAX_DEPTH]`: TThe maximum number of (player) turns to look ahead. 
        * `num_games`: The number of games for the AI to play.