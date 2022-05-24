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

FEATURE_TUPLES = [
   ('max', np.max),
   ('min', np.min),
   ('mean', np.mean),
   ('std', np.std),
   ('skew', scipy.stats.skew),
   ('kurt', scipy.stats.kurtosis),
   ('perc05', lambda x: np.percentile(x, 5)),
   ('perc25', lambda x: np.percentile(x, 25)),
   ('perc50', lambda x: np.percentile(x, 50)),
   ('perc75', lambda x: np.percentile(x, 75)),
   ('perc95', lambda x: np.percentile(x, 95))
]

def engineer_features(node, input_dir, output_dir, window_length, trim_data):
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
    
    # Set up new column names
    edata = {}
    for col in data.columns:
        for ef in FEATURE_TUPLES:
            edata[ col + "_" + ef[0] ] = []
    
    for i in range(len(data)):
        logging.debug(f"Processing row {i}")
        for name in data.columns:
            logging.debug(f"Processing column {name}")
            start = 0
            if window_length is not None:
                start = max(0, i - window_length)
            logging.debug(f"Start indexes are {start}:{i+1} (window size {window_length})")
            tdata = data[name][start:i+1]
            for feature in FEATURE_TUPLES:
                res = feature[1](tdata)
                logging.debug(f"{name + '_' + feature[0]}: {res}")
                edata[name + "_" + feature[0]].append(res)

    with pandas.HDFStore(outpath, "w") as writer:
        writer.put('ts', pandas.DataFrame(edata, index=index))

def do_work(input_dir, output_dir, window_length, trim_data):
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
        for row in reader:
            # Copy to the output CSV
            writer.writerow(row)
            # process file...
            for node in literal_eval(row['node_ids']):
                engineer_features(node, input_dir, output_dir, window_length, trim_data)

def main():
    parser = argparse.ArgumentParser(description="Perform a train-test split for Taxonomist data along a per-job axis")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--window-length", type=int, default=None,
        help="When calculating statistics, use at most N samples")
    parser.add_argument("--trim-data", type=int, default=None,
        help="Trim this many samples from the beginning and end of each series. Default 0")
    parser.add_argument("--loglevel", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])

    args = parser.parse_args()

    if args.loglevel:
        logging.basicConfig(level=args.loglevel)
    logging.debug(args)

    do_work(args.input_dir, args.output_dir, args.window_length, args.trim_data)


if __name__=="__main__":
    main()