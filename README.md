# CONLL corpus splitter

This tool reproducibly splits CONLL corpus into train, dev and test set. For the same input parameters, this tool will return exactly the same split.

It can be given a single input file, or a folder containing multiple corpus files.
Output will be written to a single file per set, suffixing the base filename with '_train', '_test' and '_dev' for train, test and dev set, respectively.
While sample-related metadata will always be present in the output, metadata related to document and/or paragraph can be omitted.

The tool will read the input file(s) twice. Once to count the samples (i.e. sentences), and second time to process them.

The tool can also create k-fold cross-validation datasets. The example shows the structure of a 5-fold cross-validation split.

```
       |---------------------------randomized-samples-----------------------------|


  k=1  |=====test=====|------------------------train------------------------|-dev-|

  k=2  |--train-|-dev-|=====test=====|-------------------train--------------------|

  k=3  |---------train---------|-dev-|=====test=====|------------train------------|

  k=4  |-----------------train----------------|-dev-|=====test=====|----train-----|

  k=5  |-------------------------train-----------------------|-dev-|=====test=====|
```

## Usage

### Command-line tool
```
$ python3 -m conll_corpus_splitter -h
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

#### Example
```
$ python3 -m conll_corpus_splitter ~/data/reldi-normtagner-hr/reldi-normtagner-hr.conllu -o ~/data/reldi-normtagner-hr/ -t 0.2 -d 0.1

```

### Python package

The logic is implemented in the `split_corpus` function.
```
from conll_corpus_splitter import split_corpus

help(split_corpus)
Help on function split_corpus in module conll_corpus_splitter.splitter:

split_corpus(source, output_folder, test=0.3, dev=0.0, seed=None, cross_validation=False, omit_metadata=False, output_filename=None, iterator_cls=<class 'conll_corpus_splitter.splitter.CONLLCorpusIterator'>)
    Splits the corpus into train, test and dev set.
    
    Parameters
    ----------
    source : str
        Source file or folder.
    output_folder : str
        Output folder.
    test : float, optional
        Size of the test set, expressed as a decimal proportion (default is 0.3).
    dev : float, optional
        Size of the dev set, expressed as a decimal proportion (default is 0.0).
    seed : int, optional
        Value for the random seed. If not provided, number of samples will be used.
    cross_validation : bool, optional
        Flag denoting whether k-fold cross-validation sets will be created (default is False).
    omit_metadata : bool, optional
        Flag denoting whether document and/or paragraph metadata will be omitted (default is False).
    output_filename : str, optional
        Filename for the output files. If not provided, filename will be the name of the source.
    iterator_cls : class, optional
        Corpus iterator class.
    
    Returns
    ----------
    NoneType
    
    Raises
    ----------
    ValueError
        Raises an exception if there are inconsistencies in the parameters.
```

For more flexibility, the logic related to iterating over samples in the corpus was implemented in a separate class. Should there be a need for a different, more elaborate logic, it can be subclassed to implement the desired behaviour.
```
from conll_corpus_splitter import CONLLCorpusIterator

help(CONLLCorpusIterator)
Help on class CONLLCorpusIterator in module conll_corpus_splitter.splitter:

class CONLLCorpusIterator(builtins.object)
 |  Class that implements iteration logic over CONLL corpus which can be composed of multiple files.
 |  
 |  Properties
 |  ----------
 |  sample_count : int
 |      Number of samples in the corpus.
 |  
 |  Methods
 |  -------
 |  __iter__()
 |      Iterates over the samples in the corpus.
 |  
 |  Methods defined here:
 |  
 |  __init__(self, *filenames, sample_start_pattern='^#\\ssent_id\\s?=', comment_pattern='^#\\s?(?P<attr_name>[^=]+?)(?:\\s?=\\s?(?P<attr_value>.+))?$', ignore_metadata_attributes=['global.columns'])
 |      Parameters
 |      ----------
 |      filenames : tuple
 |          List of input paths.
 |      sample_start_pattern : str, optional
 |          Regex matching the beggining of sample.
 |      comment_pattern : str, optional
 |          Regex matching the comment (metadata) line.
 |      ignore_metadata_attributes : list, optional
 |          List of metadata attributes which should be ignored.
 |  
 |  __iter__(self)
 |      Iterates over the samples in the corpus.
 |      
 |      Returns
 |      ----------
 |      (str, dict)
 |          Returns 2-tuple of sample text and related metadata.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  sample_count
```
