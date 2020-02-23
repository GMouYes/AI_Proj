# UrbanPlanning
## How to run the code
### Prerequisites
#### Operating systems
We tested on the most up to dated
* Windows 10
* Mac OS
* Linux Ubuntu 18.04
Users are safe to run our code on these platforms. For other platforms, generally speaking, it should also be fine but we cannot fully guarantee as no test was executed.
#### Language
To run the code you will need to install `python3`. Any version no earlier than `V3.5` will be fine.
#### Dependency
You would also need the following packages/libraries. Usually they are already automatically integrated together with python3. 
```
numpy
sys
heapq
time
random
copy
math
collections
```
### Command line
Open terminal and change directory to the Queens folder, then run the following command
```python3
python3 UrbanPlanning.py [mapFileName] [GA|HC]
```
For detailed explanation of each argv, you can simply run the following command:
```python3
python3 UrbanPlanning.py
```
## Execution Results
Results will be generated to an outputfile automatically. The following information will be shown:
* The score for this map
* At what time that score was first achieved
* The map, with the various industrial, commercial, and residential sites marked.

