from cve_bin_tool.log import LOGGER
from cve_bin_tool.cvedb import CVEDB
from cve_bin_tool.parsers.parse import valid_files as cbt_valid_files
import os
import subprocess
import argparse
from typing import List
import json

"""
This script should speed up the scanning for specific programming-language
package vulnerabilities with cve-bin-tool.

This script doe the following:
1. Walks the extracted root filesystem of the container searching for package
metadata files which cve-bin-tool would normally look for (`valid_files`).

2. Uses cve-bin-tool's language parsing functions to extract the `vendor`, 
`product` and `version` from the metadata files for each installed package.

3. Saves a CSV with the columns `vendor`, `product` and `version`.

4. Calls cve-bin-tool with the `--input-file` (`-i`) argument pointing to the
CSV file. This will do a direct database lookup for each product, vendor and
version, rather than scanning the iamge itself. The output is a JSON file
containing all CVEs for those installed packages.
"""

CVE_DIR = os.getenv("CVE_DIR", os.getcwd())


def list_language_files() -> tuple[dict[str, list[str]], dict[str, list[type]]]:
    """
    Return a dictionaries mapping language names to files which cve-bin-tool checks
    and a dictionary mapping file names to the cve-bin-tool checkers.
    
    Returns
    =======
    language_files: dict[str, list[str]]
        A dictionary mapping language names to lists of file names that cve-bin-tool checks.
    file_checkers: dict[str, list[type]]
        A dictionary mapping file names to lists of cve-bin-tool parser classes that check those files
    """

    # file_checkers is a dict mapping file names to the cve-bin-tool checkers
    file_checkers = cbt_valid_files.copy()
    file_checkers["METADATA"] = file_checkers["METADATA: "]
    file_checkers["PKG-INFO"] = file_checkers["PKG-INFO: "]


    # This is a dict mapping language name to files which cve-bin-tool checks
    language_files = {}
    for key, value in file_checkers.items():
        # get the name of the language from the module name
        # e.g. "cve_bin_tool.parsers.python.PythonParser" -> "python"
        module_name = value[0].__module__
        language = module_name.split(".")[-1]
        if language not in language_files:
            language_files[language] = []
        language_files[language].append(key)

    return language_files, file_checkers


def find_files(root_dir: str) -> list[str]:
    """
    Find all language files that CVE-bin-tool scans

    Inputs
    ======
    root_dir: str
        The root directory to search for language files

    Returns
    =======
    matches: list[str]
        A list of paths to metadata files for language packages
    """

    matches = {}
    language_files, file_checkers = list_language_files()
    for language in language_files:
        matches[language] = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename in language_files[language] and filename != "requirements.txt":
                    matches[language].append((f"{dirpath}/{filename}", file_checkers[filename]))
    return matches


def write_language_csvs(rootfs: str, language_files: List[str]) -> None:
    """
    Write a CSV file of all language packages found

    Inputs
    ======
    rootfs: str
        The path to the root filesystem of a container
    """
    cve_db = CVEDB()
    logger = LOGGER.getChild("Fuzz")

    print(f"Collecting product, vendor and version information from language files...")
    for language, file_list in language_files.items():
        if len(file_list) > 0:
            print(f"Found {len(file_list)} Language Files ({language})")

            with open(f"{CVE_DIR}/lang-{language}-packages.csv", "w") as f:
                f.write("vendor,product,version\n")
                for file, parserclslist in file_list:
                    for parsercls in parserclslist:
                        parser = parsercls(cve_db, logger)

                        output = parser.run_checker(file)
                        for out in output:
                            items = (out.product_info.vendor, out.product_info.product, out.product_info.version)
                            f.write(f"{','.join(items)}\n")

                print(f"Saved lang-{language}-packages.csv")

    # cve_db does unusual things when it exists, so let's catch it
    try:
        del cve_db
    except Exception:
        pass


def run_language_scan(language: str) -> str:
    """
    Scan the CVE database using the list of language packages found and save the
    results to a JSON file.

    Inputs
    =======
    language: str
        The name of the programming language to scan (e.g., "python", "ruby", "nodejs", etc.)
    
    Returns
    =======
    str
        The path to the JSON file containing the CVE scan results for the language packages.
        If the file does not exist, an empty string is returned.
    """

    print(f"Scanning Language Packages ({language})...")
    fname = f"{CVE_DIR}/lang-{language}-packages.csv"
    if not os.path.exists(fname):
        print(f"File {fname} does not exist. Skipping scan for {language}.")
        return ""
    cmd = [
        "cve-bin-tool",
        "-u", "never",       # Never update the local CVE database
        "-f", "json",        # Output format: JSON
        "-o", f"{CVE_DIR}/cve-bin-tool-lang-{language}-summary.json",   # Write JSON results to this file
        "-i", fname
    ]
    _ = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # do not try to do anything clever with the result.returncode here, because
    # it only ever returns 0 if there are 0 vulnerabilities!
    return fname


def scan_languages(languages) -> None:
    """
    Scan all languages found in the root filesystem for CVEs.

    Inputs
    =======
    languages: list[str]
        A list of programming languages to scan (e.g., ["python", "ruby", "nodejs", ...])

    """

    cve_files = []
    for language in languages:
        cve_file = run_language_scan(language)
        if cve_file:
            cve_files.append(cve_file)

    # collect all the results into a single file
    print("Collecting all results into a single file...")
    cve_data = []
    for file in cve_files:
        with open(file, "r") as f:
            data = json.load(f)
            cve_data.extend(data)

    with open(f"{CVE_DIR}/cve-bin-tool-lang-summary.json", "w") as f:
        json.dump(cve_data, f, indent=2)
        

def main(rootfs: str) -> None:
    """
    Scan the root filesystem for CVEs in the language packages.
    """
    language_files = find_files(rootfs)
    write_language_csvs(rootfs, language_files)
    scan_languages(list(language_files.keys()))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("rootfs", type=str)
    args = parser.parse_args()

    main(args.rootfs)
