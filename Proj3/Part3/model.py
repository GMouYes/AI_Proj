#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 21:41:02 2019

@author: mgy
"""

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import train_test_split as TTS, GridSearchCV as GCV

import numpy as np
import pickle
import random
import itertools
import multiprocessing as mp
import math
import time

seed = 1

# load data
def loadAndSplit(X,y):
    # load data and preprocess it
    y = np.ravel(y)
    
    # by default it is shuffled
    X_train, X_test, y_train, y_test = TTS(X, y, test_size = 0.2, random_state = seed)
    return X_train, X_test, y_train, y_test

# train data and save the model
def trainAndSave(model, X_train, y_train):
    model.fit(X_train, y_train)
    #y_predict = model.predict(X_test)
    return model

def simplifiedTest(target,parameters, X_train,X_test,y_train,y_test,outputResult):
    model = target(**parameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    F1 = f1_score(y_test, y_pred, average='weighted')
    #return (parameters,F1)
    outputResult.put((parameters,F1))

def myParallel(model,parameter,X_train,X_valid,y_train,y_valid,new_paras):
    parallelProcesses = 16
# =============================================================================
#     pool = mp.Pool(processes=20)
#     argList = []
#     for parameter in new_paras:
#         argList.append({'target':model,'parameters':parameter,'X_train':X_train,'X_test':X_valid,'y_train':y_train,'y_test':y_valid})
#     print(len(argList))
#     results = pool.map(simplifiedTest, argList)
# =============================================================================
    
    periods = math.ceil(len(new_paras)/parallelProcesses)
    results = []
    for i in range(periods):
        #print('fetching lists')
        newList = new_paras[parallelProcesses*i:parallelProcesses*(i+1)]
        #print('creating queues')
        # Define an output queue
        output = mp.Queue()
        # Setup a list of processes that we want to run
        processes = [mp.Process(target=simplifiedTest, \
                                args=(model,parameter,X_train,X_valid,y_train,y_valid, output)) for parameter in newList]
        #print('starting jobs')
        # Run processes
        for p in processes:
            p.start()
        #print('collecting jobs')
        # Exit the completed processes
        for p in processes:
            p.join()
        #print('appending results')
        # Get process results from the output queue
        results += [output.get() for p in processes]
        #print('take a nap')
        time.sleep(10)
        
        
    scores = [item[1] for item in results]
    myIndex = scores.index(max(scores))
    return results[myIndex][0]

def simplifiedCV(target, parameters,X_train,X_valid,y_train,y_valid):
    new_paras = [dict(zip(parameters.keys(),v)) for v in itertools.product(*parameters.values())]
    result = myParallel(target,new_paras,X_train,X_valid,y_train,y_valid,new_paras)
    return result


def RandomForest(X_train, X_test, y_train, y_test):
    modelName = 'RandomForest'
    grid_N = [10*i for i in range(5, 31)]
    parameters = {'n_estimators':grid_N, 'criterion': ('entropy', 'gini'), 'max_depth':[None], \
                  'min_samples_split':[2],'min_samples_leaf':[1], 'min_weight_fraction_leaf':[0.0], \
                  'max_features':['auto'], 'max_leaf_nodes':[None],\
                  'min_impurity_decrease':[0.0], 'min_impurity_split':[None],'bootstrap':[True], \
                  'oob_score':[False], 'n_jobs':[1],'random_state': [seed], 'verbose':[0], \
                  'warm_start':[False], 'class_weight':['balanced']}
    #RF = RandomForestClassifier()
    #clf = GCV(RF, parameters, cv=10, n_jobs=-1)
    parameter = simplifiedCV(RandomForestClassifier,parameters,X_train,X_test,y_train,y_test)
    clf = RandomForestClassifier(**parameter)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test])
    model = trainAndSave(clf, X, y)
    # print(modelName + ' is done! Cheers!')
    return model

# this is a function for debugging
def countData(y):
    count0 = 0
    count1 = 0
    count2 = 0
    for item in y:
        if item == 0:
            count0 += 1
        if item == 1:
            count1 += 1
        if item == 2:
            count2 += 1
    print('0: {}\t1: {}\t2: {}'.format(count0,count1,count2))
    
# the main function to call
def buildModel(X,y):
    # manually fix the seed to prevent any possible problem
    np.random.seed(seed)    
    random.seed(seed)
    
    X_train, X_test, y_train, y_test = loadAndSplit(X,y)
    # outputfile = 'TradAndLIWC91.txt'
    
    #countData(y_train)
    #countData(y_test)
    model = RandomForest(X_train, X_test, y_train, y_test)
    
    return model
    
# if __name__ == '__main__':
#     main()
