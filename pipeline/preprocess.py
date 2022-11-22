import argparse

from src.parsing import parse_raw_data_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Path to raw data directory, containing folders with source code', required=True)
    parser.add_argument('-o', help='Path to save collected methods and their names', required=True)
    parser.add_argument('--splits', help='Name of the splits', default=['training', 'validation', 'test'], nargs='+', type=str)
    args = parser.parse_args()

    parse_raw_data_dir(args.i, args.splits, args.o)
