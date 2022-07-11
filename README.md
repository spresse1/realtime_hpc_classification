Extract all the data, assuming you start in the base directory of the repository
```shell
$ mkdir -p Datasets/Taxonomist
$ cd Datasets/Taxonomist
$ # download the taxonomist zip to the current directory
$ wget 'https://springernature.figshare.com/ndownloader/files/11748680' -O data.zip
$ unzip data.zip 
Archive:  data.zip
   creating: data/
  inflating: data/timeseries.tar.bz2  
  inflating: data/features.hdf       
  inflating: data/metadata.csv  
$ cd data/
$ tar xf timeseries.tar.bz2
$ cd ../../..
```

Reduce the data:

```shell
$ taxonomist/reduce.py Datasets/Taxonomist/data/metadata.csv Datasets/Taxonomist/data/timeseries/ Datasets/Taxonomist/reduced
```

Scale and impute the data:

```shell
$ taxonomist/scaleimpute.py Datasets/Taxonomist/data/metadata.csv Datasets/Taxonomist/reduced/ Datasets/Taxonomist/scaledimputed
```

Then split the data for train/test (and delete the merely reduced data):

```shell
$ taxonomist/traintestsplit.py Datasets/Taxonomist/data/metadata.csv Datasets/Taxonomist/scaledimputed Datasets/Taxonomist/processed-minmax
$ rm -rf Datasets/Taxonomist/reduced
$ rm -rf Datasets/Taxonomist/scaledimputed
```

Note that this creates new metadata files in each of the train/validate 
directories that contains only the time series in that directory.

# Creating Taxonomist-equivalent Engineered Features

```shell
$ mkdir Datasets/Taxonomist/engineered/
$ taxonomist/engineerfeatures.py --trim-data 40 --only-last Datasets/Taxonomist/processed/train Datasets/Taxonomist/engineered/train-last
$ taxonomist/engineerfeatures.py --trim-data 40 --only-last Datasets/Taxonomist/processed/validate Datasets/Taxonomist/engineered/validate-last
```

# Creating Engineered Features

So far, this process has simply created a train/validate split of the data. We
also want to do some feature engineering, in order to have data comparable
to the way Taxonomist did theirs.

So, let us now create engineered features:

For 40s to match the taxonomist paper:
```shell
$ # cumulative data
$ taxonomist/engineerfeatures.py --trim-data 40 Datasets/Taxonomist/processed/validate Datasets/Taxonomist/engineered/validate-trim40/
$ taxonomist/engineerfeatures.py --trim-data 40 Datasets/Taxonomist/processed/train Datasets/Taxonomist/engineered/train-trim40/
$ # And windowed data
$ taxonomist/engineerfeatures.py --window-length 40 --trim-data 40 Datasets/Taxonomist/processed/validate Datasets/Taxonomist/engineered/validate-trim40-win40
$ taxonomist/engineerfeatures.py --window-length 40 --trim-data 40 Datasets/Taxonomist/processed/train Datasets/Taxonomist/engineered/train-trim40-win40
```

These take about 15 minutes to run in total.

# Calculate results

Next, we'll calculate the actual results. Let's create directories to put the
results in:

```shell
# For taxonomist-comparable results
$ mkdir -p results/final/lastengineered
# For cumulatively-calculated results
$ mkdir -p results/final/cumuengineered
# For rolling window results
$ mkdir -p results/final/allengineered
```

## Taxonomist-equivalent results

First, the results that are equivalent to Taxonomist's classification of entire jobs, based on the statistics over the entire job.

```shell
$ taxonomist/pointintime.py --output_dir ./results/final/lastengineered decision_tree Datasets/Taxonomist/engineered/train-last/ Datasets/Taxonomist/engineered/validate-last/
$ taxonomist/pointintime.py --output_dir ./results/final/lastengineered extra_trees Datasets/Taxonomist/engineered/train-last/ Datasets/Taxonomist/engineered/validate-last/
$ taxonomist/pointintime.py --output_dir ./results/final/lastengineered random_forest Datasets/Taxonomist/engineered/train-last/ Datasets/Taxonomist/engineered/validate-last/
$ taxonomist/pointintime.py --output_dir ./results/final/lastengineered linear_svc Datasets/Taxonomist/engineered/train-last/ Datasets/Taxonomist/engineered/validate-last/
$ taxonomist/pointintime.py --output_dir ./results/final/lastengineered svc Datasets/Taxonomist/engineered/train-last/ Datasets/Taxonomist/engineered/validate-last/
```

Results will be printed to the console as well as placed in a file in the 
directory named in the `--output_dir` parameter.

## With Cumulatively-engineered Features

```shell
$ taxonomist/pointintime.py --loglevel INFO --output_dir ./results/final/cumuengineered decision_tree Datasets/Taxonomist/engineered/train-trim40/ Datasets/Taxonomist/engineered/validate-trim40/
$ taxonomist/pointintime.py --loglevel INFO --output_dir ./results/final/cumuengineered extra_trees Datasets/Taxonomist/engineered/train-trim40/ Datasets/Taxonomist/engineered/validate-trim40/
$ taxonomist/pointintime.py --loglevel INFO --output_dir ./results/final/cumuengineered random_forest Datasets/Taxonomist/engineered/train-trim40/ Datasets/Taxonomist/engineered/validate-trim40/
```

## With Rolling-window Features

```shell
$ taxonomist/pointintime.py --loglevel INFO --output_dir ./results/final/allengineered decision_tree Datasets/Taxonomist/engineered/train-trim40-win40/ Datasets/Taxonomist/engineered/validate-trim40-win40/
$ taxonomist/pointintime.py --output_dir ./results/final/allengineered extra_trees Datasets/Taxonomist/engineered/train-trim40-win40/ Datasets/Taxonomist/engineered/validate-trim40-win40/
$ taxonomist/pointintime.py --output_dir ./results/final/allengineered random_forest Datasets/Taxonomist/engineered/train-trim40-win40/ Datasets/Taxonomist/engineered/validate-trim40-win40/
```

# Appendix: Running hyperparameter optimization

Hyperparameter optimization is much more complex than simply re-running the 
experiments using pre-optimized values. In general, hyperparameter optimization
is in `taxonomist/pointintime_hyperparams.py`. This tool relies heavily on the
[Optuna](https://optuna.readthedocs.io/en/stable/) hyperparameter optimization 
engine.

Optuna has multiple options for data storage, from in-memory to sqlite to
full MySQL/Postgres/etc databases -- anyhting supported by 
[SQLAlchemy](https://docs.sqlalchemy.org/en/14/core/engines.html).

As a user, you are on your own for database configuration. The default for the 
tool is to use an in-memory database. Doing so will result in the best set of
hyperparameters being printed at the end, but does not allow any analysis, nor
does it allow using multiple processes to determine optimal hyperparameters.
Therefore, if you are able to provide a database connection, I highly 
recommend doing so.

The general format for the hyperparameter optimization script is:

```shell
$ pointintime_hyperparams.py [--output_dir OUTPUT_DIR] [--loglevel {CRITICAL,ERROR,WARNING,INFO,DEBUG}] [--database DATABASE] [--timeout TIMEOUT] [--trials TRIALS] [--samples SAMPLES] [--trim-data TRIM_DATA] study_name {decision_tree,random_forest,extra_trees,svc,linear_svc} {grid,random,nsgaii,tpe} input_dir
```

Where:

* `study_name` is a required name for the study and is used to store the results in the database and for creating results files.
* `{decision_tree,random_forest,extra_trees,svc,linear_svc}` controls which classifier is being optimizied.
* `{grid,random,nsgaii,tpe}` controls which algorithm is used to perform optimization.
* `input_dir` is the directory which contains input data.
* `OUTPUT_DIR` is an optional directory to output various results files into. The script will automatically create a subdirectory for the job inside this directory.
* `--loglevel` controls how much to output. Few messages are printed about the `INFO` level.
* `DATABASE` is an optional SQLAlchemy string describing where to store data.
* `--timeout` is an optional limit after which no new hyperparameter optimization attempts will be run (but ongoing ones may continue to run)
* `--trials` is an optional limit for the number of hyperparameter optimization attempts to run
* `--samples` is an optional number of input samples to limit the optimization attempts to using. This can be useful to make individual trainings run more quickly, at the expense of possible accuracy. Note thr full dataset still must be loaded before the extra is discarded, as the selection of which samples to keep is done in a balanced manner.
* `--trim-data` optionally trims `TRIM_DATA` seconds off of the start and end of each job in the input data.

I recommend specifying at least one of `--timeout` or `--trials`. Some 
algorithms will run forever without some type of limitation added to them.

An example use of the hyperparameter optimization might look like this:

```shell
$ taxonomist/pointintime_hyperparams.py --timeout 27000 --database "postgresql+pg8000://optuna:optunapassword@localhost/optuna" svc_grid_study svc grid Datasets/Taxonomist/engineered/train-last/
```

Finally, it is possible to adjust the bounds on the hyperparameter searches.
However, it requires modifying the code. Starting around line 73 of the file,
there is an array named `TESTS`. Hyperparameter bounds may be adjusted in this
array. Array entries look like this:

```python
    "decision_tree": {
        TEST_OUTNAME: "decision_tree",
        TEST_CLASSIFIER: DecisionTreeClassifier,
        TEST_PARAMS: {},
        TEST_HYPERPARAMS_FUNCTION: hps_decision_tree,
        TEST_HYPERPARAMS_RANGES: {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": [ "best", "random" ],
            "max_features": ["auto", "sqrt", "log2"],
        }
    },
```

Any hyperparameter bound changes made in `TEST_HYPERPARAMS_RANGES` must also 
be reflected in the function named in `TEST_HYPERPARAMS_FUNCTION`, higher up
in the file.