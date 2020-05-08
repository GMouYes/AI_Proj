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

`python __main__.py [-h|--help] [--AI_type] {heuristic} ...`, where
* `-h|--help`: Displays command help
* `--AI_type`: If supplied, a valid AI type and the associated parameters must be supplied; else, the game starts
normally, with full human control. Valid types are:
    * `heuristic`: Uses a naive heuristic-based AI to play games. Possible arguments are
    `... heuristic [-h|--help] [num_games]`:
        * `-h|--help`: Displays command help
        * `num_games`: The number of games for the AI to play. The default is 10.