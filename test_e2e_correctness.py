#!/usr/bin/env python3

import os
import subprocess
import argparse

WORK_DIRECTORY = os.getcwd()
E2E_CORRECTNESS_DIRECTORY = f"{WORK_DIRECTORY}/e2e_correctness"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Test MAGE E2E correctness."
    )
    parser.add_argument(
        "-k", help="Add filter to tests you want to run", type=str, required=False
    )
    args = parser.parse_args()
    return args


#################################################
#                End to end tests               #
#################################################


def main(test_filter: str = None):
    os.environ["PYTHONPATH"] = E2E_CORRECTNESS_DIRECTORY
    os.chdir(E2E_CORRECTNESS_DIRECTORY)
    command = ["python3", "-m", "pytest", ".", "-vv"]
    if test_filter:
        command.extend(["-k", test_filter])
    subprocess.run(command)


if __name__ == "__main__":
    args = parse_arguments()
    test_filter = args.k
    main(test_filter=test_filter)
