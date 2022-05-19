#! /usr/bin/env python3

from matplotlib.pyplot import axis
import pandas
import logging
import argparse
import csv
import os
from ast import literal_eval
from pprint import pprint
import sklearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def get_file_data(filename, input_dir):
    filepath = os.path.join(input_dir, filename + ".hdf")
    logging.debug(f"Reading file {filepath}")
    return pandas.read_hdf(filepath)

def impute_scale_file(filename, input_dir, output_dir, imputer:SimpleImputer, scaler:StandardScaler):
    filepath = os.path.join(input_dir, filename + ".hdf")
    logging.debug(f"Handling file {filepath}")
    data = pandas.read_hdf(filepath)
    data = imputer.transform(data)
    data = scaler.transform(data)

    outpath = os.path.join(output_dir, filename + ".hdf")
    logging.info(f"Writing data to {outpath}")
    with pandas.HDFStore(outpath) as writer:
        writer.put('ts', pandas.DataFrame(data))

def do_work(label_file, timeseries_dir, output_dir):

    if not os.path.exists(output_dir):
        logging.info(f"Creating output directory {output_dir}")
        os.mkdir(output_dir)

    # Place to store loaded data
    alldata = []

    logging.info("Fetching data")
    with open(label_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            logging.debug(row)
            for node in literal_eval(row['node_ids']):
                logging.debug(f"Node id is {node}")
                alldata += [ get_file_data(node, timeseries_dir) ]
    concated = pandas.concat(alldata)

    logging.info("Training imputer")
    imputer = SimpleImputer(strategy='mean', copy=False)
    # We want the imputed values available for the scaler's training, even if
    #  this means we do it twice
    concated = imputer.fit_transform(concated)
    logging.info("Training scaler")
    scaler = StandardScaler()
    scaler.fit(concated)

    logging.info("Appling imputer and scaler")
    # Loop over all files, apply both to each, then write the file out.
    with open(label_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            logging.debug(row)
            logging.debug(row['node_ids'])
            for node in literal_eval(row['node_ids']):
                logging.debug(f"Node id is {node}")
                impute_scale_file(node, timeseries_dir, output_dir, imputer, scaler)
    
    logging.info("Done")

def main():
    parser = argparse.ArgumentParser(description="Reduce Taxonomist data to only that that would be available via out-of-band management")
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