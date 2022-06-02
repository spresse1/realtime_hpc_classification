#! /usr/bin/env python3

from ast import literal_eval
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas
import argparse
import logging
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample

CONFIDENCE_THRESHOLD=0.75

TEST_CLASSIFIER="classifier"
TEST_PARAMS="params"
TEST_OUTNAME="name"
TESTS = {
    # "knn": {
    #     TEST_OUTNAME: "knn",
    #     TEST_CLASSIFIER: KNeighborsClassifier,
    #     TEST_PARAMS: {"n_neighbors": 5},
    # },
    # "neural": {
    #     TEST_OUTNAME: "neural",
    #     TEST_CLASSIFIER: MLPClassifier,
    #     TEST_PARAMS: { "hidden_layer_sizes": (192, 192, 192, )}
    # },
    # "decision_tree": {
    #     TEST_OUTNAME: "decision_tree",
    #     TEST_CLASSIFIER: DecisionTreeClassifier,
    #     TEST_PARAMS: {}
    # },
    # "bagging-tree": {
    #     TEST_OUTNAME: "bagging-tree",
    #     TEST_CLASSIFIER: BaggingClassifier,
    #     TEST_PARAMS: {}
    # },
    "random_forest": {
        TEST_OUTNAME: "random_forest",
        TEST_CLASSIFIER: RandomForestClassifier,
        TEST_PARAMS: {}
    },
    "extra_trees": {
        TEST_OUTNAME: "extra_forest",
        TEST_CLASSIFIER: ExtraTreesClassifier,
        TEST_PARAMS: {}
    },
    "svc": {
        TEST_OUTNAME: "svc",
        TEST_CLASSIFIER: SVC,
        TEST_PARAMS: { 
            "probability": True
        },
    },
    "linear_svc": {
        TEST_OUTNAME: "linear_svc",
        TEST_CLASSIFIER: SVC,
        TEST_PARAMS: { 
            "kernel": "linear",
            "probability": True,
        },
    },
}

USE_COLS = []

def get_data(train_dir, val_dir, train_samples=None, validate_samples=None,
        trim_data=0):
    alldata = []

    logging.info("Fetching training data")
    train_label_file = os.path.join(train_dir, "metadata.csv")
    with open(train_label_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            logging.debug(row)
            for node in literal_eval(row['node_ids']):
                label = row['app']
                logging.debug(f"Node id is {node}, label {label}")

                filepath = os.path.join(train_dir, node + ".hdf")
                logging.debug(f"Reading file {filepath}")
                data = pandas.read_hdf(filepath)
                data.insert(0, "label", [ label ] * data.shape[0])
                if trim_data!=0:
                    data = data[trim_data:-trim_data]
                alldata += [ data ]

    traindata = pandas.concat(alldata)

    alldata = []

    logging.info("Fetching validation data")
    val_label_file = os.path.join(val_dir, "metadata.csv")
    with open(val_label_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            logging.debug(row)
            for node in literal_eval(row['node_ids']):
                label = row['app']
                logging.debug(f"Node id is {node}, label {label}")

                filepath = os.path.join(val_dir, node + ".hdf")
                logging.debug(f"Reading file {filepath}")
                data = pandas.read_hdf(filepath)
                data.insert(0, "label", [ label ] * data.shape[0])
                if trim_data!=0:
                    data = data[trim_data:-trim_data]
                alldata += [ data ]

    valdata = pandas.concat(alldata)

    if train_samples is not None:
        # Resample, while preserving labels
        traindata = resample(traindata, stratify=traindata.iloc[:, 0].values,
            n_samples=train_samples, random_state=42)

    if validate_samples is not None:
        # Resample, while preserving labels
        valdata = resample(valdata, stratify=valdata.iloc[:, 0].values,
            n_samples=validate_samples, random_state=42)

    return traindata.iloc[:,1:], valdata.iloc[:,1:], traindata.iloc[:,0], valdata.iloc[:,0]

def train(classifier, params, X, y,):
    classifier = classifier(**params)
    classifier.fit(X, y)
    return classifier

def classify(X_test, classifier):
    return classifier.predict(X_test)

def report(outdir, test_name, y_test, y_pred, classifier):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(os.path.join(outdir, test_name)):
        os.mkdir(os.path.join(outdir, test_name))
    
    classes = np.concatenate([classifier.classes_, [ "unknown" ]])
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print(cm)
    cr = classification_report(y_test, y_pred,labels=classifier.classes_, target_names=classifier.classes_)
    print(cr)
    with open(os.path.join(outdir, test_name, "results.txt"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")
        f.write(str(cm))
        f.write("\n")
        f.write(cr)


    fig = plt.figure(figsize=(8,8), dpi=100)
    NORMALIZE='all'
    if hasattr(ConfusionMatrixDisplay, "from_predictions"):
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, 
            labels=classes, include_values=True, 
            xticks_rotation="vertical", values_format='0.3f', normalize=NORMALIZE,
            ax=fig.add_axes([0.15,0.05,0.85,0.9],autoscale_on=True))
    else:
        cm = confusion_matrix(y_test, y_pred, labels=classes, normalize=NORMALIZE)
        disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    
    plt.savefig(os.path.join(outdir, test_name, "confusion_matrix.png"))

    if isinstance(classifier, DecisionTreeClassifier):
        fig = plt.figure(figsize=(100,100), dpi=100)
        ax = fig.add_axes([0.15,0.05,0.85,0.9],autoscale_on=True)
        sklearn.tree.plot_tree(classifier, ax=ax)
        plt.savefig(os.path.join(outdir, test_name, "decision_tree.png"))
        sklearn.tree.export_graphviz(classifier, os.path.join(outdir, test_name, "tree.dot"))
    #plt.show()

def do_work(classifier, train_dir, val_dir, out_dir, 
    train_samples=None, validate_samples=None, trim_data=0):
    test = TESTS[classifier]

    logging.info("Fetching data")
    X_train, X_val, y_train, y_val = get_data(
        train_dir, 
        val_dir,
        train_samples=train_samples, validate_samples=validate_samples, 
        trim_data=trim_data)

    logging.info("Running training")
    classifier = train(
        test[TEST_CLASSIFIER], test[TEST_PARAMS],X_train, y_train
    )

    logging.info("Making predictions")
    # Todo: use confidence intervals instead and reject unconfident decisions
    predictions = classify(X_val, classifier)
    probs = classifier.predict_proba(X_val)
    for i in range(len(probs)):
        if max(probs[i]) < CONFIDENCE_THRESHOLD:
            predictions[i] = "unknown"


    logging.info("Generating report")
    report(out_dir, test[TEST_OUTNAME], y_val, predictions, classifier)

    logging.info("Done.")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run a neural network test with antarex data")
    parser.add_argument("classifier", choices=TESTS.keys(),
        help="Which classifier to run")
    parser.add_argument("train_dir",
        help="Directory with the training data")
    parser.add_argument("validate_dir",
        help="Directory with the validation data")
    parser.add_argument("--output_dir", default="./results",
        help="Directory for results. Each classifier will make a same-named directory under this to store its results")
    parser.add_argument("--loglevel", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level to run at.")
    parser.add_argument("--train_samples", type=int, default=None,
        help="Number of training samples to use. Defaults to all. Reduction is stratified by label")
    parser.add_argument("--validate_samples", type=int, default=None,
        help="Number of validation samples to use. Defaults to all. Reduction is stratified by label")
    parser.add_argument("--trim-data", type=int, default=0,
        help="Trim this many samples from the beginning and end of each series. Default 0")

    args = parser.parse_args()

    if args.loglevel:
        logging.basicConfig(level=args.loglevel)
    logging.debug(args)

    do_work(
        args.classifier, 
        args.train_dir, 
        args.validate_dir, 
        out_dir=args.output_dir,
        trim_data=args.trim_data,
        train_samples=args.train_samples, validate_samples=args.validate_samples)