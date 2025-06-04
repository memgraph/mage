from cve_bin_tool.log import LOGGER
from cve_bin_tool.cvedb import CVEDB
from cve_bin_tool.parsers.parse import valid_files
import os
import subprocess
import argparse


def find_files(root_dir):
    """
    Find all language files that CVE-bin-tool scans    
    """
    global valid_files
    valid_files["METADATA"] = valid_files["METADATA: "]
    valid_files["PKG-INFO"] = valid_files["PKG-INFO: "]

    matches = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in valid_files and filename != "requirements.txt":
                matches.append((f"{dirpath}/{filename}",valid_files[filename]))
    return matches


def write_package_csv(rootfs):
    cve_db = CVEDB()
    logger = LOGGER.getChild("Fuzz")

    files = find_files(rootfs)
    print(f"Found {len(files)} Language Files")

    with open("lang-packages.csv", "w") as f:
        f.write("vendor,product,version\n")
        for file, parserclslist in files:
            for parsercls in parserclslist:
                parser = parsercls(cve_db, logger)

                output = parser.run_checker(file)
                for out in output:
                    items = (out.product_info.vendor, out.product_info.product, out.product_info.version)
                    f.write(f"{','.join(items)}\n")

    print("Saved lang-packages.csv")

    # cve_db does unusual things when it exists, so let's catch it
    try:
        del cve_db
    except Exception:
        pass


def run_scan():

    print("Scanning Language Packages")
    cmd = [
        "cve-bin-tool",
        "-u", "never",       # Never update the local CVE database
        "-f", "json",        # Output format: JSON
        "-o", "cve-bin-tool-lang-summary.json",   # Write JSON results to this file
        "-i", "lang-packages.csv"
    ]
    _ = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


def main(rootfs):

    write_package_csv(rootfs)
    run_scan()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("rootfs", type=str)
    args = parser.parse_args()

    main(args.rootfs)
