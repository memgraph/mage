from aggregate_build_tests import list_daily_release_packages
import os
import argparse


def main() -> None:
    """
    return the relevant image URL to be scanned for CVEs
    """
    date = int(os.getenv("CURRENT_BUILD_DATE"))

    parser = argparse.ArgumentParser()
    parser.add_argument("arch", type=str)
    args = parser.parse_args()

    # translate to dict key
    key, arch = ("Docker (arm64)", "arm64") if args.arch == "arm64" else ("Docker (x86_64)", "x86_64")

    packages = list_daily_release_packages(date)
    url = packages[key][arch]

    print(url)


if __name__ == "__main__":
    main()
