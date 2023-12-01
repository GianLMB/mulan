"""Train Mulan model on custom data. Not implemented yet!"""

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(
        prog="train",
        description=__doc__,
    )
    args = parser.parse_args()
    return args

def main():
    get_args()
    raise NotImplementedError("This script has not been implemented yet.")

if __name__ == "__main__":
    main()

