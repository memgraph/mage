import requests
import os
from typing import List, Tuple
import re
from collections import defaultdict
import json


def read_summary_file(filename: str) -> List[str] | dict:

    found = os.path.isfile(filename)
    if not found:
        print(f"{filename} summary not found")
        return None

    with open(filename, "r") as f:
        if filename.endswith(".json"):
            data = json.load(f)
        else:
            data = f.readlines()
        return data


def _cbt_read_cve_summary(lines: List[str]) -> dict:
    """
    Fetches the vulnerability summary
    """
    result = {}
    in_table = False

    for line in lines:
        # Start parsing after encountering the CVE SUMMARY header
        if 'CVE SUMMARY' in line:
            in_table = True
            continue

        if in_table:
            # End parsing when the bottom border is reached
            if line.strip().startswith('└'):
                break

            # Process only data rows
            if line.strip().startswith('│'):
                # Skip the header row
                if 'Severity' in line:
                    continue

                match = re.match(r'│\s*(\w+)\s*│\s*(\d+)\s*│', line)
                if match:
                    severity = match.group(1)
                    count = int(match.group(2))
                    result[severity] = count

    return result


def _cbt_read_cpe_summary(lines: List[str]) -> dict:
    """
    Parses a list of strings containing a CPE Summary ASCII table and returns
    a dictionary keyed by product, where each value is a dict of the other fields.
    """
    # Define the columns in the order they appear in the table
    cols = [
        'Vendor',
        'Product',
        'Version',
        'Latest Upstream Stable Version',
        'CRITICAL CVEs Count',
        'HIGH CVEs Count',
        'MEDIUM CVEs Count',
        'LOW CVEs Count',
        'UNKNOWN CVEs Count',
        'TOTAL CVEs Count'
    ]

    result = {}
    in_table = False

    for line in lines:
        # Start parsing once we see the CPE SUMMARY header
        if 'CPE SUMMARY' in line:
            in_table = True
            continue

        if not in_table:
            continue

        # Stop when we hit the bottom border of the table
        if line.strip().startswith('└'):
            break

        # Only process data rows (which start with '│')
        if line.strip().startswith('│'):
            # Use split on '│' to grab all cell contents
            cells = line.split('│')[1:-1]
            # Strip whitespace from each cell
            cells = [c.strip() for c in cells]
            # If the row has exactly the right number of columns, map them
            if len(cells) == len(cols):
                row = dict(zip(cols, cells))
                product = row.pop('Product')
                # Convert numeric fields from strings to int when possible
                for key in row:
                    # only convert the CVE counts and version stays string
                    if key.endswith('Count'):
                        row[key] = int(row[key])
                result[product] = row

    return result


def _cbt_newfound_cves(lines: List[str]) -> dict:
    """
    Parses a list of strings containing a NewFound CVEs ASCII table and returns
    a dict keyed by product, where each value is a list of dicts with the fields:
      Vendor, Version, CVE Number, Source, Severity, Score (CVSS Version)
    """
    cols = [
        'Vendor',
        'Product',
        'Version',
        'CVE Number',
        'Source',
        'Severity',
        'Score (CVSS Version)'
    ]

    result = defaultdict(list)
    in_table = False

    for line in lines:
        # start when we see the header
        if 'NewFound CVEs' in line:
            in_table = True
            continue
        if not in_table:
            continue

        # stop at bottom border
        if line.strip().startswith('└'):
            break

        # only process rows beginning with '│'
        if line.strip().startswith('│'):
            cells = [c.strip() for c in line.split('│')[1:-1]]
            if len(cells) == len(cols):
                row = dict(zip(cols, cells))
                # group by product
                prod = row.pop('Product')
                result[prod].append(row)

    return dict(result)


def parse_cbt_summary() -> dict:
    """
    This will read in the `cve-bin-tool-summary.txt` and scan for detected
    vulnerabilities.
    """

    lines = read_summary_file("cve-bin-tool-summary.txt")
    if not lines:
        return {}

    cve_summary = _cbt_read_cve_summary(lines)
    cpe_summary = _cbt_read_cpe_summary(lines)
    newfound_cve = _cbt_newfound_cves(lines)

    return {
        "summary": cve_summary,
        "cpe": cpe_summary,
        "cve": newfound_cve
    }


def parse_grype_summary() -> dict:
    """
    This will read in the `grype-summary.txt`
    """

    data = read_summary_file("grype-summary.json")
    if not data:
        return {}

    results = data["matches"]
    cves = {}
    for result in results:
        fixed_versions = result["vulnerability"]["fix"]
        if len(fixed_versions) > 0:
            fixed = ",".join(fixed_versions)
        else:
            fixed = "unfixed"
        cves[result["artifact"]["name"]] = {
            "id": result["vulnerability"]["id"],
            "status": result["vulnerability"]["fix"]["state"],
            "severity": result["vulnerability"]["severity"],
            "version": result["artifact"]["version"],
            "fixed": fixed
        }

    summary = {}
    for _, cve in cves.items():
        severity = cve["severity"]
        if severity not in summary:
            summary[severity] = 0
        summary[severity] += 1

    return {
        "summary": summary,
        "cve": cves
    }


def parse_trivy_summary() -> dict:
    """
    This will read in the `trivy-summary.json`
    """

    data = read_summary_file("trivy-summary.json")
    if not data:
        return {}

    results = data["Results"]
    cves = {}
    for result in results:
        if result["Class"] in ["os-pkgs", "lang-pkgs"]:
            for vuln in result["Vulnerabilities"]:
                cves[vuln["PkgName"]] = {
                    "id": vuln["VulnerabilityID"],
                    "status": vuln["Status"],
                    "severity": vuln["Severity"],
                    "version": vuln["InstalledVersion"],
                    "fixed": vuln.get("FixedVersion", "unfixed")
                }

    summary = {}
    for _, cve in cves.items():
        severity = cve["severity"]
        if severity not in summary:
            summary[severity] = 0
        summary[severity] += 1

    return {
        "summary": summary,
        "cve": cves
    }


def combine_summaries(cbt: dict, grype: dict, trivy: dict) -> Tuple[dict, str]:

    cbt = {k.lower(): v for k, v in cbt.items() if v > 0}
    grype = {k.lower(): v for k, v in grype.items() if v > 0}
    trivy = {k.lower(): v for k, v in trivy.items() if v > 0}

    summary = {}
    for dct in [cbt, grype, trivy]:
        dct = {k.lower(): v for k, v in dct.items() if v > 0}

        for k, v in dct.items():
            if k not in summary:
                summary[k] = 0
            summary[k] += v

    keys = ["negligible", "low", "medium", "high", "critical"]
    emojis = [":grinning_face_with_star_eyes:", ":grinning:", ":sweat_smile:", ":melting_face:", ":mushroom_cloud:"]

    # build a little table
    msg = "```\n"
    msg += "| Severity   | Counts |\n"
    msg += "|------------|--------|\n"
    for key in keys:
        msg += f"| {key:10s} | {summary.get(key, 0):6} |\n"
    msg += "\n```\n"

    # find worst
    emoji = emojis[0]
    for i, key in enumerate(keys):
        if summary.get(key, 0) > 0:
            emoji = emojis[i]

    msg += f"Overal Status: {emoji}\n"

    return summary, msg


def format_slack_table(items):
    """
    Given a list of dicts with keys: Package, Version, Severity, CVE,
    return a string containing a code-block table with evenly padded columns.
    """
    if not items:
        return "```Nothing to see here...```"

    # 1) Define your columns and compute the max width for each
    headers = ["Package", "Version", "Severity", "CVE"]
    widths = {}
    for h in headers:
        widths[h] = max(
            len(h),
            *(len(str(item.get(h, ""))) for item in items)
        )

    # 2) Build the header row and a separator
    header_row = " | ".join(h.ljust(widths[h]) for h in headers)
    sep_row = "-+-".join("-" * widths[h] for h in headers)

    # 3) Build each data row
    data_rows = []
    for item in items:
        row = " | ".join(str(item.get(h, "")).ljust(widths[h]) for h in headers)
        data_rows.append(row)

    # 4) Wrap it all in a code block
    table = "\n".join([header_row, sep_row] + data_rows)
    return f"```\n{table}\n```"


def cbt_message(cbt):

    if not cbt:
        return ""

    msg = "CVE-bin-tool Summary:\n"
    msg += "============================"
    items = []
    for package in cbt["cve"]:
        for item in cbt["cve"][package]:
            if item["Severity"] in ["MEDIUM", "HIGH", "CRITICAL"]:
                items.append({
                    "Package": package,
                    "Version": item["Version"],
                    "Severity": item["Severity"],
                    "CVE": item["CVE Number"]
                })

    table = format_slack_table(items)
    return msg + f"\n{table}\n"


def grype_trivy_message(cves, name):

    if not cves:
        return ""

    msg = f"{name} Summary:\n"
    msg += "============================"
    items = []
    for package, item in cves["cve"].items():
        if item["severity"].upper() in ["MEDIUM", "HIGH", "CRITICAL"]:
            items.append({
                "Package": package,
                "Version": item["version"],
                "Severity": item["severity"],
                "CVE": item["id"]
            })

    table = format_slack_table(items)
    return msg + f"\n{table}\n"


def create_slack_message(cbt, grype, trivy) -> str:

    msg_start = "Vulnerability Scan Results...\n\n"

    # combine summaries
    _, summary_msg = combine_summaries(
        cbt.get("summary", {}),
        grype.get("summary", {}),
        trivy.get("summary", {}),
    )

    cbt_msg = cbt_message(cbt)
    grype_msg = grype_trivy_message(grype, "Grype")
    trivy_msg = grype_trivy_message(trivy, "Trivy")

    msg = "\n".join([msg_start, summary_msg, grype_msg, trivy_msg, cbt_msg])
    return msg


def post_message(msg):

    url = os.getenv("INFRA_WEBHOOK_URL")
    try:
        response = requests.post(
            url,
            json={"text": msg}
        )
    except Exception:
        print(f"Response: {response.status_code}")


def main():

    # collect results
    cbt = parse_cbt_summary()
    grype = parse_grype_summary()
    trivy = parse_trivy_summary()

    msg = create_slack_message(cbt, grype, trivy)
    post_message(msg)
    print(msg)


if __name__ == "__main__":
    main()

