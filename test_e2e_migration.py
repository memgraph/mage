#!/usr/bin/env python3

import os
import subprocess
import argparse
import sys

WORK_DIRECTORY = os.getcwd()
E2E_MIGRATION_DIRECTORY = f"{WORK_DIRECTORY}/e2e_migration"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test MAGE E2E migration.")
    parser.add_argument(
        "-k",
        help="Filter tests by database type (e.g., 'mysql', 'postgresql')",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    return args


def main(test_filter: str = None):
    """Main function to run e2e migration tests."""
    # Use the run_migration_tests.py script which handles docker-compose automatically
    os.environ["PYTHONPATH"] = E2E_MIGRATION_DIRECTORY
    os.chdir(E2E_MIGRATION_DIRECTORY)
    command = ["python3", "run_migration_tests.py"]
    if test_filter:
        command.extend(["-k", test_filter])

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    args = parse_arguments()
    main(test_filter=args.k)
