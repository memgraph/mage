import sys
import csv
import argparse


def read_all_csv_from_stdin():
    return list(csv.DictReader(sys.stdin))


def validate_first_as_int(data, field, expected_value):
    assert len(data) == 1
    assert int(data[0][field]) == int(
        expected_value
    ), f"Got {data[0][field]}, expected {expected_value}."
    print(f"Validation of the first {field} is OK.")


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Smoke Tests Validator",
    )
    subparsers = parser.add_subparsers(
        help="sub-command help", dest="action", required=True
    )

    first_as_int_args_parser = subparsers.add_parser(
        "first_as_int", help="Validate first return value as integer"
    )
    first_as_int_args_parser.add_argument(
        "-f", "--field", help="Name of the field to test", required=True
    )
    first_as_int_args_parser.add_argument(
        "-e", "--expected", help="Expected value", required=True
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_arguments()

    data = read_all_csv_from_stdin()

    if args.action == "first_as_int":
        validate_first_as_int(data, args.field, args.expected)
