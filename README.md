# CONLL corpus splitter

This tool reproducibly splits CONLL corpus into train, dev and test set. For the same input parameters, this tool will return exactly the same split.

It can be given a single input file, or a folder containing multiple corpus files.
Output will be written to a single file per set, suffixing the base filename with '_train', '_test' and '_dev' for train, test and dev set, respectively.
While sample-related metadata will always be present in the output, metadata related to document and/or paragraph can be omitted.

The tool will read the input file(s) twice. Once to count the samples (i.e. sentences), and second time to process them.

The tool can also create k-fold cross-validation datasets. The example shows the structure of a 5-fold cross-validation split.

```
    |---------------------------randomized-samples-----------------------------|


    |=====test=====|------------------------train------------------------|-dev-|

    |--train-|-dev-|=====test=====|-------------------train--------------------|

    |---------train---------|-dev-|=====test=====|------------train------------|

    |-----------------train----------------|-dev-|=====test=====|----train-----|

    |-------------------------train-----------------------|-dev-|=====test=====|
```

## Usage

### Command-line tool
```
$ python3 splitter.py -h
usage: CONLL corpus splitter [-h] [-o OUTPUT_FOLDER] [-t TEST] [-d DEV]
                             [-s SEED] [-f OUTPUT_FILENAME]
                             [--cross-validation] [--omit-metadata]
                             source

Reproducibly splits CONLL corpus into train, dev and test set.

positional arguments:
  source                Path to the source file/folder.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FOLDER      Path to the output folder. (default: None)
  -t TEST, --test TEST  Test set size. (default: 0.3)
  -d DEV, --dev DEV     Dev set size. (default: 0.0)
  -s SEED, --seed SEED  Manually set random seed. (default: None)
  -f OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
                        Specify the filename for output files. (default: None)
  --cross-validation    Create k-fold cross-validation datasets. (default:
                        False)
  --omit-metadata       Do not write document and/or paragraph metadata to
                        output files. (default: True)
```

