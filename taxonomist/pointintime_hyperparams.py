#! /usr/bin/env python3

from ast import literal_eval
import csv
import os
import signal
import sys
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import optuna
import pandas
import argparse
import logging
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

CONFIDENCE_THRESHOLD=0.75

TEST_CLASSIFIER="classifier"
TEST_PARAMS="params"
TEST_OUTNAME="name"

# A function that returns an aarray of hyperparameter values to use for a trial
TEST_HYPERPARAMS_FUNCTION="hyperparam_function"

# Names and valid ranges for each hyperparamaeter. Needed for grid search
TEST_HYPERPARAMS_RANGES="hyperparam_ranges"

def hps_rbf_svc(trial):
    ret = {
        "C": trial.suggest_float("C", 0.000001, 10.0e6),
        "tol": trial.suggest_float("tol", 1e-6, 1e-1, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
    }
    return ret

def hps_linear_svc(trial):
    ret = {
        "C": trial.suggest_float("C", 0.000001, 10.0e6),
        "tol": trial.suggest_float("tol", 1e-6, 1e-1, log=True)
    }
    return ret

def hps_random_forest(trial:optuna.trial):
    ret = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    }
    return ret

TESTS = {
    "random_forest": {
        TEST_OUTNAME: "random_forest",
        TEST_CLASSIFIER: RandomForestClassifier,
        TEST_PARAMS: {},
        TEST_HYPERPARAMS_FUNCTION: hps_random_forest,
        TEST_HYPERPARAMS_RANGES: {
            "n_estimators": range(50, 500, 5),
            "criterion": ["gini", "entropy", "log_loss"]
        }
    },
    "extra_trees": {
        TEST_OUTNAME: "extra_forest",
        TEST_CLASSIFIER: ExtraTreesClassifier,
        TEST_PARAMS: {},
        TEST_HYPERPARAMS_FUNCTION: hps_random_forest,
        TEST_HYPERPARAMS_RANGES: {
            "n_estimators": range(50, 500, 5),
            "criterion": ["gini", "entropy", "log_loss"]
        }
    },
    "svc": {
        TEST_OUTNAME: "svc",
        TEST_CLASSIFIER: SVC,
        TEST_PARAMS: { 
            "kernel": "rbf",
            "probability": True
        },
        TEST_HYPERPARAMS_FUNCTION: hps_rbf_svc,
        TEST_HYPERPARAMS_RANGES: {
            "C": np.logspace(-6, 6, 100),
            "tol": np.logspace(-6, -1),
            "gamma": ["scale", "auto"],
        },
    },
    "linear_svc": {
        TEST_OUTNAME: "linear_svc",
        TEST_CLASSIFIER: SVC,
        TEST_PARAMS: { 
            "kernel": "linear",
            "probability": True,
        },
        TEST_HYPERPARAMS_FUNCTION: hps_linear_svc,
        TEST_HYPERPARAMS_RANGES: {
            "C": np.logspace(-6, 6, 100),
            "tol": np.logspace(-6, -1),
        },
    },
}

SAMPLERS = {
    "grid": optuna.samplers.GridSampler,
    "random": optuna.samplers.RandomSampler,
    "nsgaii": optuna.samplers.NSGAIISampler,
    "tpe": optuna.samplers.TPESampler,
}


USE_COLS = []

class SignalHandler:
    def __init__(self, study) -> None:
        self.study = study

    def handle_usr1(self, signalnum, frame):
        print("Caught signal, terminating as jobs finish")
        self.study.stop()

def override_predict(self, X):
    predictions = super(self.__class__, self).predict(X)
    probs = self.predict_proba(X)
    for i in range(len(probs)):
        if max(probs[i]) < CONFIDENCE_THRESHOLD:
            predictions[i] = "unknown"
    return predictions

def override_score(self, X, y, sample_weight=None):
    """
    Overrides the score function of Classifiers to allow forcing results with
    low probability into the unknown category.
    """
    logging.debug("Score called")
    predictions = self.predict(X)

    # These two lines borrowed from the sklearn definition:
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L650
    from sklearn.metrics import accuracy_score
    return accuracy_score(y, predictions, sample_weight=sample_weight)

class StudyRunner:
    def __init__(self, input_dir, hyperparam_func, classifier, classifier_args=None, samples=None, trim_data=0):
        self.classifier = classifier
        self.hpfunc = hyperparam_func
        self.classifier_args = classifier_args

        logging.info("Fetching data")
        self.X, self.y = self.get_data(
            input_dir, 
            samples=samples,
            trim_data=trim_data)
        logging.info("Done fetching data")

    def get_data(self, input_dir, samples=None, trim_data=0):
        alldata = []

        train_label_file = os.path.join(input_dir, "metadata.csv")
        with open(train_label_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                logging.debug(row)
                for node in literal_eval(row['node_ids']):
                    label = row['app']
                    logging.debug(f"Node id is {node}, label {label}")

                    filepath = os.path.join(input_dir, node + ".hdf")
                    logging.debug(f"Reading file {filepath}")
                    data = pandas.read_hdf(filepath)
                    data.insert(0, "label", [ label ] * data.shape[0])
                    if trim_data!=0:
                        data = data[trim_data:-trim_data]
                    alldata += [ data ]

        data = pandas.concat(alldata)

        if samples is not None:
            # Resample, while preserving labels
            data = resample(data, stratify=data.iloc[:, 0].values,
                n_samples=samples, random_state=42)

        return data.iloc[:,1:], data.iloc[:,0]

    def get_classifier(self, trial):
        params = {}
        if self.hpfunc is not None:
            params = self.hpfunc(trial)

        cparams = params
        if self.classifier_args is not None:
            cparams = { **cparams, **self.classifier_args }

        # Make a dynamic class with the classier as superclass
        DuckClassifier = type('DuckClassifier', (self.classifier,), {})
        # Override the score function to use ours that 
        DuckClassifier.score = override_score
        DuckClassifier.predict = override_predict

        return DuckClassifier(**cparams)

    def objective(self,trial):
        logging.info("Running objective")

        classifier = self.get_classifier(trial)

        # logging.debug("Inserting duck-type predict function")
        # # override the predict function so that scoring makes sense
        # # duck-type hack this on.
        # classifier.score = override_score

        logging.debug("Running cross-validation")
        return cross_val_score(classifier, self.X, self.y, cv=5).mean()

def report(outdir, test_name, studyrunner, trial):
    classifier = studyrunner.get_classifier(trial)
    # X_train, X_test, y_train, y_test = train_test_split(studyrunner.X, studyrunner.y)
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    y_pred = cross_val_predict(classifier, studyrunner.X, studyrunner.y)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(os.path.join(outdir, test_name)):
        os.mkdir(os.path.join(outdir, test_name))
    
    classes = np.unique(studyrunner.y)
    unkclasses = np.concatenate([classes, [ "unknown" ]])
    cm = confusion_matrix(studyrunner.y, y_pred, labels=unkclasses)
    print(cm)
    cr = classification_report(studyrunner.y, y_pred, labels=classes, target_names=classes)
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
        disp = ConfusionMatrixDisplay.from_predictions(studyrunner.y, y_pred, 
            labels=classes, include_values=True, 
            xticks_rotation="vertical", values_format='0.3f', normalize=NORMALIZE,
            ax=fig.add_axes([0.15,0.05,0.85,0.9],autoscale_on=True))
    else:
        cm = confusion_matrix(studyrunner.y, y_pred, labels=classes, normalize=NORMALIZE)
        disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    
    plt.savefig(os.path.join(outdir, test_name, "confusion_matrix.png"))

    if isinstance(classifier, DecisionTreeClassifier):
        fig = plt.figure(figsize=(100,100), dpi=100)
        ax = fig.add_axes([0.15,0.05,0.85,0.9],autoscale_on=True)
        sklearn.tree.plot_tree(classifier, ax=ax)
        plt.savefig(os.path.join(outdir, test_name, "decision_tree.png"))
        sklearn.tree.export_graphviz(classifier, os.path.join(outdir, test_name, "tree.dot"))
    #plt.show()

def do_work(study_name, classifier, sampler, input_dir, out_dir, db_string=None, timeout=None, trials=None,
    samples=None, trim_data=0):
    classifier_data = TESTS[classifier]

    params = {}

    # Grid search needs to know the range of the grid
    if sampler == "grid":
        params = { **params, **{"search_space": classifier_data[TEST_HYPERPARAMS_RANGES]} }
    # elif sampler == "nsgaii":
    #     params = { **params, **{"population_size": 25} }

    logging.info(f"Creating study, sampler args: {params}")
    study = optuna.create_study(
        study_name=study_name, load_if_exists=True, storage=db_string,
        direction="maximize",
        # Build the specified sampler with the specified ranges
        sampler=SAMPLERS[sampler](**params),
    )

    sh = SignalHandler(study)
    signal.signal(signal.SIGUSR1, sh.handle_usr1)

    obj = StudyRunner(
        input_dir, 
        classifier_data[TEST_HYPERPARAMS_FUNCTION], 
        classifier_data[TEST_CLASSIFIER],
        classifier_data[TEST_PARAMS],
        samples=samples,
        trim_data=trim_data
    )

    logging.info("Running study")
    study.optimize(obj.objective, show_progress_bar=True,
        # timeout=3600,
        timeout=timeout,
        n_trials=trials,
    )
    # Check for other running trials. If none, write image and wrap up
    running_trials = study.get_trials(states=[optuna.trial.TrialState.RUNNING])
    if len(running_trials)==0:
        logging.info("Writing graph images related to the study")
        if not os.path.exists("output_images"):
            os.mkdir("output_images")
        if not os.path.exists(os.path.join("output_images", study_name)):
            os.mkdir(os.path.join("output_images", study_name))
        optuna.visualization.plot_contour(study).write_image(os.path.join("output_images", study_name, "contour.png"))
        optuna.visualization.plot_edf(study).write_image(os.path.join("output_images", study_name, "edf.png"))
        optuna.visualization.plot_intermediate_values(study).write_image(os.path.join("output_images", study_name, "intermediate_values.png"))
        optuna.visualization.plot_optimization_history(study).write_image(os.path.join("output_images", study_name, "optimization_history.png"))
        optuna.visualization.plot_parallel_coordinate(study).write_image(os.path.join("output_images", study_name, "parallel_coordinates.png"))
        optuna.visualization.plot_param_importances(study).write_image(os.path.join("output_images", study_name, "param_importances.png"))
        # optuna.visualization.plot_pareto_front(study).write_image(os.path.join("output_images", study_name, "pareto_front.png"))
        optuna.visualization.plot_slice(study).write_image(os.path.join("output_images", study_name, "slice.png"))

        logging.info("Rerunning best trial and writing graphics")

        report(out_dir, study_name, obj, study.best_trial)
    logging.info("All done!")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run a neural network test with antarex data")
    parser.add_argument("study_name")
    parser.add_argument("classifier", choices=TESTS.keys(),
        help="Which classifier to run")
    parser.add_argument("search_sampler", choices=SAMPLERS.keys())
    parser.add_argument("input_dir",
        help="Directory with the training data")
    parser.add_argument("--output_dir", default="./results",
        help="Directory for results. Each classifier will make a same-named directory under this to store its results")
    parser.add_argument("--loglevel", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level to run at.")
    parser.add_argument("--database", required=False)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--samples", type=int, default=None,
        help="Number of training samples to use. Defaults to all. Reduction is stratified by label")
    parser.add_argument("--trim-data", type=int, default=0,
        help="Trim this many samples from the beginning and end of each series. Default 0")

    args = parser.parse_args()

    if args.loglevel:
        logging.basicConfig(level=args.loglevel)
    logging.debug(args)

    do_work(
        args.study_name, 
        args.classifier, 
        args.search_sampler,
        args.input_dir, 
        out_dir=args.output_dir,
        db_string=args.database,
        timeout=args.timeout,
        trials=args.trials,
        samples=args.samples,
        trim_data=args.trim_data,
    )