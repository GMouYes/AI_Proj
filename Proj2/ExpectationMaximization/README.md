# EM for soft clustering
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
copy
scipy
matplotlib
```
### Command line
Open terminal and change directory to the Queens folder, then run the following command
```shell
python3 EM.py [dataFileName] [#clusters]
```
For detailed explanation of each argv, you can simply run the following command:
```shell
python3 EM.py
```
## Execution Results
Results will be printed out automatically. The following information will be shown:
* Your input choice
* Best fitting clusters' information: means, covariances and weights
* Total log-likelihood
* Model's BIC
* Cost of time to reach the best solution
* Number of restarts taken to reach the best solution
