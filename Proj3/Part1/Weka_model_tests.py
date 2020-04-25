import os
import pandas as pd
import openpyxl
import sklearn.linear_model
import javabridge
from collections import defaultdict
import itertools
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
import weka.core.typeconv as typeconv


def test_classifier(dataset: Instances, classifier: Classifier, params: dict):
    vars = params.keys()
    vals = params.values()

    results = defaultdict(list)

    for val_combo in itertools.product(*vals):
        results["numInstances"].append(dataset.num_instances)
        results["numAttributes"].append(dataset.num_attributes)
        opts = dict(zip(vars, val_combo))

        for opt in opts:
            results[opt].append(opts[opt])
            classifier.set_property(opt, opts[opt] if not isinstance(opts[opt], float) else typeconv.double_to_float(
                opts[opt]))

        evl = Evaluation(dataset)
        classifier.build_classifier(dataset)
        evl.test_model(classifier, dataset)
        results["Training_Accuracy"].append(evl.percent_correct)
        results["size"].append(int(javabridge.call(classifier.jobject, "measureTreeSize", "()D")))
        evl.crossvalidate_model(classifier, dataset, 10, Random(1))
        results["CV_Accuracy"].append(evl.percent_correct)

    return results


def linear_regression(class_test_results: dict):
    df = pd.DataFrame(class_test_results)
    X = df.loc[:, "numInstances":"size"]
    y = df.loc[:, "CV_Accuracy"]

    reg = sklearn.linear_model.LinearRegression().fit(X, y)
    print("Dependent variables:")
    print(X.columns.values)

    print("Model intercept:", reg.intercept_)
    print("Model coefficients:", reg.coef_)

    df['Regression_Prediction'] = reg.predict(X)
    df['Regression_Error'] = df['Regression_Prediction'] - df['CV_Accuracy']
    return df


def write_to_excel(fit_results: pd.DataFrame, fname: str, sheet_name: str):
    if os.path.isfile(fname):
        book = openpyxl.load_workbook(fname)
        w = pd.ExcelWriter(fname, engine='openpyxl', mode='a')
        w.book = book
        w.sheets = dict((ws.title, ws) for ws in book.worksheets)
    else:
        w = pd.ExcelWriter(fname, engine='openpyxl')
    with w as writer:
        fit_results.to_excel(writer, sheet_name)


def main():
    jvm.start()

    data_dir = r'C:\Program Files\Weka-3-8-4\data'

    datasets = ['breast-cancer.arff', 'credit-g.arff']

    outfile = "Modeling CV Accuracy.xlsx"
    loader = Loader()

    # col_template = ["numInstances", "numAttributes", "binarySplits", "collapseTree", "doNotMakeSplitPointActualValue",
    #                 "minNumObj", "useLaplace", "useMDLcorrection", "Training_Accuracy","size", "CV_Accuracy"]

    dataset_results = defaultdict(list)

    for datafile in datasets:
        dataset = loader.load_file(os.path.join(data_dir, datafile))
        dataset.class_is_last()

        # Three possibilities with different parameter sets:
        # 1. reducedErrorPruning = False, unpruned = False
        # 2. reducedErrorPruning = True, unpruned = False
        # 3. reducedErrorPruning = False, unpruned = True

        param_template = {"binarySplits": [True, False], "collapseTree": [True, False],
                          "doNotMakeSplitPointActualValue": [True, False],
                          "minNumObj": [*range(1, 6), *range(10, 101, 10)], "useLaplace": [True, False],
                          "useMDLcorrection": [True, False]}
        # 1.
        classifier = Classifier(".J48")
        params = param_template.copy()
        params.update({"confidenceFactor": [x * 0.1 for x in range(1, 6)]})

        sheet_name = datafile.split('.')[0] + " rEP=F,unp=F"
        print("Modeling", sheet_name)

        eval_results = test_classifier(dataset, classifier, params)
        dataset_results[datafile].append(eval_results)
        fit_results = linear_regression(eval_results)

        write_to_excel(fit_results, outfile, sheet_name)

        # 2.
        classifier = Classifier(".J48")
        classifier.set_property("reducedErrorPruning", True)

        params = param_template.copy()
        params.update({"numFolds": [*range(2, 11)]})

        sheet_name = datafile.split('.')[0] + " rEP=T,unp=F"
        print("Modeling", sheet_name)

        eval_results = test_classifier(dataset, classifier, params)
        dataset_results[datafile].append(eval_results)
        fit_results = linear_regression(eval_results)

        write_to_excel(fit_results, outfile, sheet_name)

        # 3.
        classifier = Classifier(".J48")
        classifier.set_property("reducedErrorPruning", False)
        classifier.set_property("unpruned", True)

        params = param_template.copy()

        sheet_name = datafile.split('.')[0] + " rEP=F,unp=T"
        print("Modeling", sheet_name)

        eval_results = test_classifier(dataset, classifier, params)
        dataset_results[datafile].append(eval_results)
        fit_results = linear_regression(eval_results)

        write_to_excel(fit_results, outfile, sheet_name)

    # Make combined model for all datasets
    sheet_names = ["combined rEP=F,unp=F", "combined rEP=T,"
                   "unp=F", "combined rEP=F,unp=T"]
    for i in range(len(list(dataset_results.values())[0])):
        combined_results = defaultdict(list)
        for datafile in datasets:
            for key in dataset_results[datafile][i]:
                combined_results[key] += dataset_results[datafile][i][key]

        print("Modeling", sheet_names[i])
        fit_results = linear_regression(combined_results)
        write_to_excel(fit_results, outfile, sheet_names[i])

    jvm.stop()


if __name__ == '__main__':
    main()
