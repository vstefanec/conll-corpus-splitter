import math
import os
import random
import re

from collections import OrderedDict
from contextlib import ExitStack

from .utils import Dataset, RotatingList, MetadataValue, MetadataDiffDict


COMMENT_PATTERN = r'^#\s?(?P<attr_name>[^=]+?)(?:\s?=\s?(?P<attr_value>.+))?$'


class CONLLCorpusIterator(object):
    def __init__(self, *filenames, sample_start_pattern=r'^#\ssent_id\s?=',
                 comment_pattern=COMMENT_PATTERN):
        self.filenames = filenames
        self.comment_pattern = comment_pattern
        self.sample_start_pattern = sample_start_pattern
        self._sample_count = None

    def __iter__(self):
        with ExitStack() as stack:
            files = [stack.enter_context(open(filename, 'r')) for filename in self.filenames]
            line_index = -1
            for file in files:
                text_buffer = ''
                metadata = MetadataDiffDict()
                reading_sample = False
                for line in file:
                    line_index += 1
                    if not line.strip() and reading_sample:
                        # end of sample
                        yield text_buffer, metadata.copy()
                        reading_sample = False
                        text_buffer = ''
                        metadata = MetadataDiffDict()
                    elif not line.strip():
                        # empty line
                        continue
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
                            metadata[groups['attr_name']] = MetadataValue(
                                groups['attr_value'],
                                m.string[m.start():m.end()],
                                line_index
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

    test = float(test)
    dev = float(dev)

    if (test + dev) >= 1:
        raise ValueError('Test set and dev set percentage together are greater than 1.')

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
            else:
                raise RuntimeError()

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

                out_files[fold][destination].write('{}\n'.format(sample))

    print('Done.')
