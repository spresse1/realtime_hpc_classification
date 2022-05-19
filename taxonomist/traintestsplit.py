#! /usr/bin/env python3

import argparse
from ast import literal_eval
import logging
import os
import csv
from sklearn.model_selection import train_test_split
from shutil import copy2

def do_work(labelfile, timeseries_dir, output_dir):
    if not os.path.exists(output_dir):
        logging.info(f"Making output directory {output_dir}")
        os.mkdir(output_dir)
    traindir = os.path.join(output_dir, "train")
    validatedir = os.path.join(output_dir, "validate")
    if not os.path.exists(traindir):
        os.mkdir(traindir)
    if not os.path.exists(validatedir):
        os.mkdir(validatedir)

    labels = []
    files = []
    with open(labelfile) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row['app'])
            files.append({
                "datafiles": literal_eval(row['node_ids']),
                "label": row["app"],
                "run_id": row["run_id"],
            })
    
    X_train, X_test, y_train, y_test = train_test_split(
        files, labels, stratify=labels, test_size=0.2, random_state=42)

    # Now copy files where they belong...
    for items in X_train:
        logging.info(f"Info moving {items}")
        for file in items["datafiles"]:
            logging.info(f"Moving {file} to training set")
            copy2(
                os.path.join(timeseries_dir, file + ".hdf"),
                os.path.join(traindir, file + ".hdf"),
            )
    for items in X_test:
        logging.info(f"Info moving {items}")
        for file in items["datafiles"]:
            logging.info(f"Moving {file} to validation set")
            copy2(
                os.path.join(timeseries_dir, file + ".hdf"),
                os.path.join(validatedir, file + ".hdf"),
            )


def main():
    parser = argparse.ArgumentParser(description="Perform a train-test split for Taxonomist data along a per-job axis")
    parser.add_argument("labels")
    parser.add_argument("timeseries_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--loglevel", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])

    args = parser.parse_args()

    if args.loglevel:
        logging.basicConfig(level=args.loglevel)
    logging.debug(args)

    do_work(args.labels, args.timeseries_dir, args.output_dir)


if __name__=="__main__":
    main()