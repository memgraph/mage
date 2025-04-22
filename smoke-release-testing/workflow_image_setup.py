import sys
import os
import re
import argparse
# add mage root dir to python path to find other functions
sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)
from aggregate_build_tests import list_daily_release_packages  # noqa: E402

# Compile regex patterns
URL_PATTERN = re.compile(
    r'^(https?://[\w\-\.]+(?:/[\w\-\./?%&=]*)?)$'
)
DATE_PATTERN = re.compile(
    r'^(?P<year>\d{4})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])$'
)
DOCKER_PATTERN = re.compile(
    r'^([a-z0-9]+(?:[._\-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._\-][a-z0-9]+)*)*):([A-Za-z0-9][A-Za-z0-9._\-]*)$'
)
VERSION_PATTERN = re.compile(
    r'^\d+\.\d+(?:\.\d+)?$'
)



def classify_string(s: str) -> str:
    """
    Classify the string provided to see if it is a URL, docker repo tag, or date.

    - [url] for URLs to files (starting with http:// or https://)
    - [date] for dates in the format yyyymmdd
    - [docker] for Docker image name:tag

    If no pattern matches, returns the original string unchanged.
    """
    if URL_PATTERN.match(s):
        return "url"
    elif DATE_PATTERN.match(s):
        return "date"
    elif DOCKER_PATTERN.match(s):
        return "docker"
    elif VERSION_PATTERN.match(s):
        return "version"
    else:
        return s


def get_daily_url(date: str, arch: str, malloc: bool) -> str:
    """
    Given a date of the format yyyymmdd, find and return the URL of the
    appropriate image.
    """

    packages = list_daily_release_packages(int(date), return_url=True)

    try:
        arch_name = "x86_64" if arch == "amd64" else "arm64"
        key = f"Docker ({arch_name})"
        key_image = f"{arch_name}-malloc" if malloc else arch_name
        url = packages[key][key_image]
    except KeyError:
        url = "fail"

    return url


def get_version_docker(version: str, malloc: str):
    """
    convert version number to docker image tag

    This will only work for 3.0 onwards, for anything else supply the URL or 
    full docker tag to the workflow
    """

    parts = [int(x) for x in version.split(".")]

    # remove patch version if == 0
    if len(parts) == 3 and parts[-1] == 0:
        version = version[:version.rfind(".")]

    # check whether version is before 3.2 or not (change in tag format)
    major, minor = parts[:2]
    if major < 3 or (major == 3 and minor < 2):
        repo_tag = f"memgraph/memgraph-mage:{version}-memgraph-{version}"
    else:
        repo_tag = f"memgraph/memgraph-mage:{version}"

    if malloc:
        repo_tag = f"{repo_tag}-malloc"

    return repo_tag


def string_to_boolean(bool_str: str) -> bool:
    """
    convert string to Boolean
    """

    return True if bool_str.lower() == "true" else False


def main() -> None:
    parser = argparse.ArgumentParser(description="Check image/url/date")

    parser.add_argument(
        "image",
        type=str,
        help="Image tag, URL or daily build date"
    )

    parser.add_argument(
        "arch",
        type=str,
        help="CPU Arch: arm64|amd64"
    )

    parser.add_argument(
        "malloc",
        type=str,
        help="Is a malloc build: true|false"
    )

    args = parser.parse_args()

    # classify image
    cls = classify_string(args.image)

    if cls == "date":
        out = get_daily_url(
            args.image,
            args.arch,
            string_to_boolean(args.malloc)
        )
        cls = "url"
    elif cls == "version":
        out = get_version_docker(args.image, string_to_boolean(args.malloc))
        cls = "docker"
    else:
        out = args.image

    print(f"{cls} {out}")


if __name__ == "__main__":
    main()
