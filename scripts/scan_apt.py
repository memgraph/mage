import numpy as np
import subprocess
import os
import json
from cve_bin_tool.log import LOGGER
from cve_bin_tool.cvedb import CVEDB
import subprocess
import argparse


def get_apt_packages(container="memgraph"):

    cmd = [
        "docker", "exec", container,
        "dpkg-query",
        "--show",
        "--showformat={\"name\": \"${binary:Package}\", \"version\": \"${Version}\"}, "
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,        # so that stdout/stderr come back as Python strings
    )
    result = result.stdout
    packages = json.loads(f"[{result[:result.rfind(',')]}]")

    print(f"Found {len(packages)} installed DEB packages")
    return packages


def get_package_vendor_pairs(cve_db, packages):
    
    return cve_db.get_vendor_product_pairs(packages)


def combine_vendor_product_version(packages, pairs):

    prod_vends = {}
    for pair in pairs:
        prod = pair["product"]
        vend = pair["vendor"]
        if not prod in prod_vends:
            prod_vends[prod] = []
        prod_vends[prod].append(vend)

    out = []
    for package in packages:
        prod = package["name"]
        ver = package["version"]

        if prod in prod_vends:
            vends = prod_vends[prod]
            for vend in vends:
                out.append((vend, prod, ver))

    return out


def save_apt_package_csv(packages):

    with open("apt-packages.csv", "w") as f:
        f.write("vendor,product,version\n")
        for package in packages:
            f.write(f"{','.join(package)}\n")


def run_scan():

    cmd = [
        "cve-bin-tool",
        "-u", "never",       # Never update the local CVE database
        "-f", "json",        # Output format: JSON
        "-o", "cve-bin-tool-apt-summary.json",   # Write JSON results to this file
        "-i", "apt-packages.csv"
    ]
    _ = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


def main(container):
    cve_db = CVEDB()

    packages = get_apt_packages(container)
    pairs = get_package_vendor_pairs(cve_db, packages)
    package_info = combine_vendor_product_version(packages, pairs)
    save_apt_package_csv(package_info)
    print(f"Checking {len(package_info)} packages with cve-bin-tool...")
    run_scan()

    # cve_db does unusual things when it exists, so let's catch it
    try:
        del cve_db
    except Exception:
        pass

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("container", type=str)
    args = parser.parse_args()

    main(args.container)
    