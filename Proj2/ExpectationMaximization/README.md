# EM for soft clustering
## Files under the folder
There are 4 python files, 6 csv files and 1 readme file in the current folder:\
Python files:
```
EM.py				where main function locates
algo.py				where algorithm locates
HandleInput.py		consumes input and interpret them
HandleOutput.py		print out cleaner format of model results
```

CSV files:
```
sample_EM_data.csv			sample data for debugging and testing
sample_EM_data_label.csv	sample data containing actual labels
test1.csv 					synthetic data created by ourselves
test_1.csv 					same data but with labels
test2.csv 					another synthetic data created by ourselves
test_2.csv 					same data but with labels
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
