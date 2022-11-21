import math
import os
import random
import re

from collections import OrderedDict
from contextlib import ExitStack

from .utils import Dataset, RotatingList, MetadataValue, MetadataDiffDict


COMMENT_PATTERN = r'^#\s?(?P<attr_name>[^=]+?)(?:\s?=\s?(?P<attr_value>.+))?$'


class CONLLCorpusIterator(object):
    """
    Class that implements iteration logic over CONLL corpus which can be composed of multiple files.

    Properties
    ----------
    sample_count : int
        Number of samples in the corpus.

    Methods
    -------
    __iter__()
        Iterates over the samples in the corpus.
    """
    def __init__(self, *filenames, sample_start_pattern=r'^#\ssent_id\s?=', sample_end_pattern=r'\n',
                 comment_pattern=COMMENT_PATTERN, ignore_metadata_attributes=['global.columns'], append_newline=True):
        """
        Parameters
        ----------
        filenames : tuple
            List of input paths.
        sample_start_pattern : str, optional
            Regex matching the beggining of sample.
        sample_end_pattern : str, optional
            Regex matching the end of sample.
        comment_pattern : str, optional
            Regex matching the comment (metadata) line.
        ignore_metadata_attributes : list, optional
            List of metadata attributes which should be ignored.
        append_newline : bool, optional
            Append newline to every sample.
        """
        self.filenames = filenames
        self.comment_pattern = comment_pattern
        self.sample_start_pattern = sample_start_pattern
        self.sample_end_pattern = sample_end_pattern
        self.ignore_metadata_attributes = ignore_metadata_attributes
        self.append_newline = append_newline
        self._sample_count = None

    def __iter__(self):
        """
        Iterates over the samples in the corpus.

        Returns
        ----------
        (str, dict)
            Returns 2-tuple of sample text and related metadata.
        """
        with ExitStack() as stack:
            files = [stack.enter_context(open(filename, 'r')) for filename in self.filenames]
            line_index = -1
            for file in files:
                text_buffer = ''
                metadata = MetadataDiffDict()
                reading_sample = False
                for line in file:
                    line_index += 1
                    if re.match(self.sample_end_pattern, line) and reading_sample:
                        # end of sample
                        if self.append_newline:
                            text_buffer += '\n'
                        yield text_buffer, metadata.copy()
                        reading_sample = False
                        text_buffer = ''
                        metadata = MetadataDiffDict()
                    elif reading_sample:
                        text_buffer += line
                    else:
                        if re.match(self.sample_start_pattern, line):
                            # start of sample
                            reading_sample = True
                            text_buffer += line
                            continue

                        m = re.match(self.comment_pattern, line)
                        if m:
                            groups = m.groupdict()
                            if groups['attr_name'] not in self.ignore_metadata_attributes:
                                metadata[groups['attr_name']] = MetadataValue(
                                    value=groups['attr_value'],
                                    text=m.string[m.start():m.end()],
                                    line_no=line_index
                                )

    def _count_samples(self):
        print('Counting samples...')
        with ExitStack() as stack:
            files = [stack.enter_context(open(filename, 'r')) for filename in self.filenames]
            sample_count = 0
            for file in files:
                for line in file:
                    if re.match(self.sample_start_pattern, line):
                        sample_count += 1
            self._sample_count = sample_count
            print('%d samples read.' % sample_count)

    @property
    def sample_count(self):
        if self._sample_count is None:
            self._count_samples()
        return self._sample_count


def split_corpus(source, output_folder, test=0.3, dev=0.0, seed=None, cross_validation=False,
                 omit_metadata=False, output_filename=None, iterator_cls=CONLLCorpusIterator):
    """
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
    """

    test = float(test)
    dev = float(dev)

    if (test + dev) >= 1:
        raise ValueError('Test and dev set proportions together are greater than 1.')

    source = os.path.normpath(source)

    if output_filename:
        if '.' in output_filename:
            output_filename = '{}.'.join(output_filename.rsplit('.', 1))
        else:
            output_filename = output_filename + '{}'

    if os.path.isdir(source):
        in_files = [
            os.path.join(source, f) for f in sorted(os.listdir(source)) if os.path.isfile(os.path.join(source, f))
        ]
        if not output_filename:
            output_filename = os.path.basename(source) + '{}'
        if not in_files:
            raise ValueError('No input files found.')
    else:
        in_files = [source]
        if not output_filename:
            output_filename = os.path.basename(source)
            if '.' in output_filename:
                output_filename = '{}.'.join(output_filename.rsplit('.', 1))
            else:
                output_filename = output_filename + '{}'

    cci = iterator_cls(*in_files)

    if seed:
        random.seed(seed)
    else:
        random.seed(cci.sample_count)

    dev_sample_count = math.floor(cci.sample_count * dev)
    test_sample_count = math.floor(cci.sample_count * test)
    train_sample_count = cci.sample_count - (dev_sample_count + test_sample_count)

    k = 1
    if cross_validation:
        k = math.floor(1 / test)

    sample_indexes = RotatingList(range(cci.sample_count))
    random.shuffle(sample_indexes)

    datafolds = []
    for fold in range(k):
        datafold = {}
        test_start_index = fold * test_sample_count
        test_end_index = test_start_index + test_sample_count
        dev_start_index = test_start_index - dev_sample_count
        dev_end_index = test_start_index
        train_start_index = test_end_index
        train_end_index = train_start_index + train_sample_count

        test_samples = sorted(sample_indexes[test_start_index:test_end_index])
        dev_samples = sorted(sample_indexes[dev_start_index:dev_end_index])
        train_samples = sorted(sample_indexes[train_start_index:train_end_index])

        for sample_index in sample_indexes:
            if sample_index in test_samples:
                datafold[sample_index] = Dataset.TEST
            elif sample_index in train_samples:
                datafold[sample_index] = Dataset.TRAIN
            elif sample_index in dev_samples:
                datafold[sample_index] = Dataset.DEV

        datafolds.append(datafold)

    for fold in range(len(datafolds)):
        datafolds[fold] = OrderedDict(sorted(datafolds[fold].items(), key=lambda x: x[0])).values()

    sample_index_relay = []
    for fold_destinations in zip(*datafolds):
        sample_index_relay.append(fold_destinations)

    with ExitStack() as stack:
        out_files = []

        for fold in range(k):
            if k == 1:
                folder = ''
            else:
                folder = str(fold+1)
                os.makedirs(os.path.join(output_folder, folder), exist_ok=True)

            out_files_fold = []
            out_files_fold.append(
                stack.enter_context(
                    open(os.path.join(output_folder, folder, output_filename.format('_train')), 'w')
                )
            )

            if dev_sample_count:
                out_files_fold.append(
                    stack.enter_context(
                        open(os.path.join(output_folder, folder, output_filename.format('_dev')), 'w')
                    )
                )
            else:
                out_files_fold.append(None)

            out_files_fold.append(
                stack.enter_context(
                    open(os.path.join(output_folder, folder, output_filename.format('_test')), 'w')
                )
            )

            out_files.append(out_files_fold)

        global_meta = MetadataDiffDict()
        fold_meta = [[MetadataDiffDict() for _ in (Dataset.TRAIN, Dataset.DEV, Dataset.TEST)] for _ in range(k)]
        for sample_index, (sample, meta) in enumerate(cci):
            global_meta.update(meta)
            for fold, destination in enumerate(sample_index_relay[sample_index]):
                diff = fold_meta[fold][destination].diff_and_update(global_meta)
                print('Sample index: {}, k={} --> {}'.format(sample_index, fold, repr(destination)))

                if not omit_metadata:
                    for _, v in diff.items():
                        out_files[fold][destination].write('{}\n'.format(v.text))

                out_files[fold][destination].write(sample)

    print('Done.')
