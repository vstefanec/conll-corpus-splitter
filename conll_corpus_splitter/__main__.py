import argparse
import os

from conll_corpus_splitter import split_corpus


def main(args):
    output_folder = args.output_folder or os.getcwd()
    split_corpus(args.source, output_folder=output_folder, test=args.test, dev=args.dev,
                 seed=args.seed, cross_validation=args.cross_validation, omit_metadata=args.omit_metadata,
                 output_filename=args.output_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='CONLL corpus splitter',
        description='Reproducibly splits CONLL corpus into train, dev and test set.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('source', help='Path to the source file/folder.')
    parser.add_argument('-o', dest='output_folder', help='Path to the output folder.')
    parser.add_argument('-t', '--test', type=float, help='Test set size.', default=0.3)
    parser.add_argument('-d', '--dev', type=float, help='Dev set size.', default=0.0)
    parser.add_argument('-s', '--seed', type=int, help='Manually set random seed.')
    parser.add_argument('-f', '--output-filename', dest='output_filename', type=str,
                        help='Specify the filename for output files.')
    parser.add_argument('--cross-validation', dest='cross_validation', action='store_true',
                        help='Create k-fold cross-validation datasets.')
    parser.add_argument('--omit-metadata', dest='omit_metadata', action='store_true',
                        help='Do not write document and/or paragraph metadata to output files.')
    args = parser.parse_args()
    main(args)
