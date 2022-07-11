#! /usr/bin/env python3

import argparse
from ast import literal_eval
from asyncore import write
import logging
import os
import csv
import numpy as np
import pandas
import scipy
from sklearn.model_selection import train_test_split
from shutil import copy2

# (Name to use in column labels, name of pandas.DataFrame function to apply, arguments to function)
ENGINEERED_FEATURES = [
   ('max', 'max', {}),
   ('min', 'min', {}),
   ('mean', 'mean', {}),
   ('std', 'std', {}),
   ('skew', 'skew', {}),
   ('kurt', 'kurt', {}),
   ('perc05', 'quantile', {'quantile': 0.05}),
   ('perc25', 'quantile', {'quantile': 0.25}),
   ('perc50', 'quantile', {'quantile': 0.50}),
   ('perc75', 'quantile', {'quantile': 0.75}),
   ('perc95', 'quantile', {'quantile': 0.95})
]

def engineer_features(node, input_dir, output_dir, window_length, trim_data, only_last=False):
    inpath = os.path.join(input_dir, node + ".hdf")
    outpath = os.path.join(output_dir, node + ".hdf")
    logging.debug(f"Handling file {inpath}")
    logging.info(f"Writing data to {outpath}")
    data = pandas.read_hdf(inpath)
    
    # grab index data
    index = data.index
    
    # Shorten index and data if we're skipping early data
    if trim_data is not None:
        logging.info(f"Trimming down data")
        index = index[trim_data:-trim_data]
        data = data[trim_data:-trim_data]
    
    feature_frames = []
    if window_length is None:
        window_length = len(data)
    for feature in ENGINEERED_FEATURES:
        trans = getattr(
            data.rolling(
                window_length, closed="both", min_periods=1
            ), feature[1])(**feature[2])
        trans.columns = [ x + "_" + feature[0] for x in trans.columns ]
        feature_frames += [ trans ]
    edata = pandas.concat(feature_frames, axis=1).fillna(0)

    if only_last:
        edata = pandas.DataFrame(edata.iloc[-1:])
        index = [ index[-1] ]
    
    with pandas.HDFStore(outpath, "w") as writer:
        writer.put('ts', pandas.DataFrame(edata, index=index))

def do_work(input_dir, output_dir, window_length, trim_data, only_last):
    if not os.path.exists(output_dir):
        logging.info(f"Making output directory {output_dir}")
        os.mkdir(output_dir)
    
    inlabelfile = os.path.join(input_dir, "metadata.csv")
    outlabelfile = os.path.join(output_dir, "metadata.csv")

    colnames = None
    with open(inlabelfile) as f, open(outlabelfile, "w") as o:
        reader = csv.DictReader(f)
        colnames = reader.fieldnames
        writer = csv.DictWriter(o, fieldnames=colnames)
        writer.writeheader()
        for row in reader:
            # Copy to the output CSV
            writer.writerow(row)
            # process file...
            for node in literal_eval(row['node_ids']):
                engineer_features(node, input_dir, output_dir, window_length, trim_data, only_last=only_last)

def main():
    parser = argparse.ArgumentParser(description="Perform a train-test split for Taxonomist data along a per-job axis")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--window-length", type=int, default=None,
        help="When calculating statistics, use at most N samples")
    parser.add_argument("--trim-data", type=int, default=None,
        help="Trim this many samples from the beginning and end of each series. Default 0")
    parser.add_argument("--only-last", action='store_true',
        help="Store only the last data point in each file (equal to taxonomist features.hdf)")
    parser.add_argument("--loglevel", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])

    args = parser.parse_args()

    if args.loglevel:
        logging.basicConfig(level=args.loglevel)
    logging.debug(args)

    do_work(args.input_dir, args.output_dir, args.window_length, args.trim_data, args.only_last)


if __name__=="__main__":
    main()