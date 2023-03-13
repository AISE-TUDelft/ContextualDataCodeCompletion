This repository contains replication code for the paper "Enriching Source Code with Contextual Data for Code Completion Models: An Empirical Study".

## Installation

Ensure you have NodeJS and Python 3 installed.
Then install dependencies:

```bash
pip install -r requirements.txt
npm install
```

## Dataset
The dataset can be retrieved from [Zenodo](https://doi.org/10.5281/zenodo.7553738). The dataset should be extracted to the `./data` folder of this repository.


## Replication

The two main files to run for replication are `create-datasets.sh` and `evaluate.sh`.

These files should be run in the root directory of the project (i.e. directly in the `Replication-Code` folder).

### `create-datasets.sh`

This file does the following:

1. Copy the dataset and add marker comments (/\*<marker:number>\*/)
2. Copy the marked dataset, install third-party dependencies, and add type annotations
3. Determine which projects had all dependencies installed succesfully
4. Copy the marked dataset, and remove all type annotations
5. Analyze the dataset (#LOC, #Files, Type Explicitness)
6. Create train/test/validation files for consumption by UniXcoder, CodeGPT, and InCoder
6.1 Note that UniXcoder and CodeGPT use the same input files. In practice, it will only show files for UniXcoder, but these are intended to be used for both UniXcoder and CodeGPT.

### `evaluate.sh`

This file does the following:

1. Post process predictions
2. Evaluate post processed predictions/computes all metrics (both for complete lines and single tokens)
3. Performs the statistical analysis

This is done for every model.

Note that this script expects a `predictions` folder to be present inside the `data` folder.
The `predictions` folder should have subfolders of the format `./data/predictions/<unixcoder|codegpt|incoder>/<normal|untyped|explicit>-<all|none|docblock|single_line|multi_line>/` and should contain the respective `test.json` file for the model & dataset, and `predictions.txt` file generated based on this `test.json` file.

## Configuration

Some parameters can be configured through `config.json`.
