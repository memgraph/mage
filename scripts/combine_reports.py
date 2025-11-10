import json
from io import StringIO
from rich.table import Table
from rich.console import Console
import argparse
import os


def read_json_file(filename):

    if not os.path.exists(filename):
        return None

    with open(filename, "r") as f:
        data = json.load(f)
    return data


def extract_grype_data(data):
    if data is None:
        return []

    matches = data["matches"]

    out = []
    for match in matches:
        fixed = match["vulnerability"]["fix"]["versions"]
        if len(fixed) > 0:
            fixed = fixed[0]
        else:
            fixed = None
        out.append({
            "package": match["artifact"]["name"],
            "version": match["artifact"]["version"],
            "vulnerabilityID": match["vulnerability"]["id"],
            "type": match["artifact"]["type"],
            "fixed": fixed,
            "severity": match["vulnerability"]["severity"].capitalize(),
        })
    return out


def extract_trivy_data(data):
    if data is None:
        return []

    out = []
    results = data["Results"]
    for result in results:
        type = result["Type"]
        matches = result["Vulnerabilities"]
        for match in matches:
            out.append({
                "package": match["PkgName"],
                "version": match["InstalledVersion"],
                "vulnerabilityID": match["VulnerabilityID"],
                "type": type.replace("python-pkg", "python"),
                "fixed": None,
                "severity": match["Severity"].capitalize(),
            })
    return out


def combine_reports(grype_data, trivy_data):
    out = grype_data.copy()

    grype_keys = [(x["package"], x["version"], x["vulnerabilityID"]) for x in grype_data]
    trivy_keys = [(x["package"], x["version"], x["vulnerabilityID"]) for x in trivy_data]

    for key, item in zip(trivy_keys, trivy_data):
        if key not in grype_keys:
            out.append(item)

    # sort by package name
    out.sort(key=lambda x: x["package"])
    return out


def format_table(data):
    table = Table(title="Vulnerabilities")
    table.add_column("Package", justify="left")
    table.add_column("Version", justify="left")
    table.add_column("VulnerabilityID", justify="left")
    table.add_column("Severity", justify="left")
    table.add_column("Type", justify="left")
    table.add_column("Fixed", justify="left")
    for item in data:
        table.add_row(
            item["package"],
            item["version"],
            item["vulnerabilityID"],
            item["severity"],
            item["type"],
            item["fixed"]
        )
    return table


def save_table_to_file(table, filename):
    console = Console(file=StringIO(), width=None, force_terminal=False)
    console.print(table)
    output = console.file.getvalue()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("grype_file", type=str)
    parser.add_argument("trivy_file", type=str)
    args = parser.parse_args()

    grype_data = extract_grype_data(read_json_file(args.grype_file))
    trivy_data = extract_trivy_data(read_json_file(args.trivy_file))
    combined_data = combine_reports(grype_data, trivy_data)
    table = format_table(combined_data)
    outdir = os.getenv("CVE_DIR", os.getcwd())
    save_table_to_file(table, f"{outdir}/combined_report.txt")


if __name__ == "__main__":
    main()
