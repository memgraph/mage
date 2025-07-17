import sys
import csv
import argparse


def read_all_csv_from_stdin():
    return list(csv.DictReader(sys.stdin))


def get_main_parser(data):
    main_addr=None
    for instance in data:
        if instance["role"] == '"main"':
            main_addr=instance["management_server"][1:-1].split(".")[0]
    print(main_addr)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Smoke Tests Validator",
    )
    subparsers = parser.add_subparsers(
        help="sub-command help", dest="action", required=True
    )
    subparsers.add_parser(
        "get_main_parser", help="Get main parser"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    data = read_all_csv_from_stdin()
    if args.action == "get_main_parser":
        get_main_parser(data)
