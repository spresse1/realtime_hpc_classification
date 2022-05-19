#! /usr/bin/env python3

from matplotlib.pyplot import axis
import pandas
import logging
import argparse
import csv
import os
from ast import literal_eval
from pprint import pprint

def reduce_file(filename, input_dir, output_dir):
    filepath = os.path.join(input_dir, filename + ".hdf")
    logging.info(f"Handling file {filepath}")
    i = pandas.read_hdf(filepath)
    
    totalJiffies = i[["idle_procstat", "sys_procstat", "guest_procstat",
        "guest_nice_procstat", "softirq_procstat", "user_procstat",
        "irq_procstat", "iowait_procstat", "nice_procstat"]].sum(axis=1)

    o = pandas.DataFrame()
    # Processor time data
    o["KernelPercent"] = i["sys_procstat"] / totalJiffies
    o["UserPercent"] = i["user_procstat"] / totalJiffies

    # Network IO Data
    o["TXUnicastFrames"] = i["AR_NIC_NETMON_ORB_EVENT_CNTR_REQ_PKTS_metric_set_nic"]

    # Memory data
    o["CacheSizeMB"] = i["Cached_meminfo"]
    o["RemainingSpareBlockPercentage"] = i["MemFree_meminfo"] / 67108864.0 #64 GB in KB

    # Pseudo-disk data
    o["DiskReadBytes"] = i["RDMA_rx_bytes_cray_aries_r"]

    # Power data
    o["PowerWatts"] = i["power(W)_cray_aries_r"]

    with pandas.HDFStore(os.path.join(output_dir, filename + ".hdf")) as writer:
        writer.put('ts', o)


def do_work(label_file, timeseries_dir, output_dir):

    if not os.path.exists(output_dir):
        logging.info(f"Creating output directory {output_dir}")
        os.mkdir(output_dir)

    logging.info("Beginning data reduction")
    with open(label_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            logging.debug(row)
            logging.debug(row['node_ids'])
            for node in literal_eval(row['node_ids']):
                logging.debug(f"Node id is {node}")
                reduce_file(node, timeseries_dir, output_dir)
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