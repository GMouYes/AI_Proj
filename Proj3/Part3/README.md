# RL Robotic Truck Driver
## Files under the folder
There are 4 python files and 1 readme file in the current folder:\
Python files:
```
expSearch.py		where main function locates
qLearning.py		where algorithm locates
handleInput.py		consumes input and interpret them
handleOutput.py		print out cleaner format of model results
```

README.md 			markdown file recording 

## How to run the program
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
time
random
pprint
itertools
collections
pandas
```

### Command line
Open terminal and change directory to the ExpectationMaximization folder, then run the following command
```shell
python3 expSearch.py [truck capacity] [road length] [penalty for start the truck] [clock ticks]
```
For detailed explanation of each argv, you can simply run the following command:
```shell
python3 expSearch.py
``` 

## Execution Results
Results will be printed out automatically. The following information will be shown:
* Training time
* Best Policy of the map
* Average reward
