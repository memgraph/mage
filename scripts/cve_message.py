import requests
import os
from typing import List, Tuple
import json
import argparse

CVE_DIR = os.getenv("CVE_DIR", os.getcwd())


def read_summary_file(filename: str) -> List[str] | dict:
    """
    Read the contents of a file and return it as a list or dict.
    """

    found = os.path.isfile(filename)
    if not found:
        print(f"{filename} summary not found")
        return None

    print(f"Found {filename}")
    with open(filename, "r") as f:
        if filename.endswith(".json"):
            data = json.load(f)
        else:
            data = f.readlines()
        return data


def severity_summary(data: List[dict]) -> dict:
    """
    count the total number of CVEs per severity level
    """
    summary = {}
    for cve in data:
        severity = cve["severity"]
        if severity not in summary:
            summary[severity] = 0
        summary[severity] += 1

    return summary


def reformat_cbt_data(data: List[dict]) -> List[dict]:
    """
    reformats the cve-bin-tool data to a common format
    """

    out = []
    for item in data:
        new = {
            "id": item["cve_number"],
            "product": item["product"],
            "severity": item["severity"].upper(),
            "fixed": None,
            "version": item["version"]
        }
        if new not in out:
            out.append(new)
    return out


def parse_cbt_summary() -> dict:
    """
    Reads cve-bin-tool summaries
    """

    parts = ["memgraph", "lang", "apt"]
    out = {}
    for part in parts:
        fname = f"{CVE_DIR}/cve-bin-tool-{part}-summary.json"
        if os.path.isfile(fname):
            print(f"Found {fname}")
            data = read_summary_file(fname)
            data = reformat_cbt_data(data)
            out[part] = {
                "summary": severity_summary(data),
                "cve": data
            }

    return out


def reformat_grype_data(data: List[dict]) -> List[dict]:
    """
    reformats the data from grype to a common format
    """

    out = []
    for item in data:
        fixed_versions = item["vulnerability"]["fix"]
        if len(fixed_versions) > 0:
            fixed = ",".join(fixed_versions)
        else:
            fixed = "unfixed"
        new = {
            "id": item["vulnerability"]["id"],
            "product": item["artifact"]["name"],
            "status": item["vulnerability"]["fix"]["state"],
            "severity": item["vulnerability"]["severity"].upper(),
            "version": item["artifact"]["version"],
            "fixed": fixed
        }
        if new not in out:
            out.append(new)
    return out


def parse_grype_summary() -> dict:
    """
    This will read in the `grype-summary.json`
    """

    data = read_summary_file(f"{CVE_DIR}/grype-summary.json")
    if not data:
        return {}

    results = data["matches"]
    cves = reformat_grype_data(results)
    summary = severity_summary(cves)

    return {
        "summary": summary,
        "cve": cves
    }


def reformat_trivy_data(data: List[dict]) -> List[dict]:
    """
    reformat the trivy JSON to a common format.
    """

    out = []
    for item in data:
        if item["Class"] in ["os-pkgs", "lang-pkgs"]:
            for vuln in item["Vulnerabilities"]:
                new = {
                    "id": vuln["VulnerabilityID"],
                    "product": vuln["PkgName"],
                    "status": vuln["Status"],
                    "severity": vuln["Severity"].upper(),
                    "version": vuln["InstalledVersion"],
                    "fixed": vuln.get("FixedVersion", "unfixed")
                }
                if new not in out:
                    out.append(new)
    return out


def parse_trivy_summary() -> dict:
    """
    This will read in the `trivy-summary.json`
    """

    data = read_summary_file(f"{CVE_DIR}/trivy-summary.json")
    if not data:
        return {}

    results = data["Results"]
    cves = reformat_trivy_data(results)
    summary = severity_summary(cves)

    return {
        "summary": summary,
        "cve": cves
    }


def combine_summaries(summary_list: List[dict]) -> Tuple[dict, str]:
    """
    This will combine the summaries from all of the tests into one.

    Inputs
    ======
    summary_list: List[dict]
        A list of dictionaries, each containing a summary for one test.

    Returns
    =======
    Tuple[dict, str]
        A tuple containing the counts of each vulnerability severity.

    """

    summary = {}
    for dct in summary_list:
        dct = {k: v for k, v in dct.items() if v > 0}
        for k, v in dct.items():
            if k not in summary:
                summary[k] = 0
            summary[k] += v

    keys = ["UNKNOWN", "NEGLIGIBLE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    emojis = [
        ":interrobang:",
        ":grinning_face_with_star_eyes:"*2,
        ":grinning:"*3,
        ":sweat_smile:"*4,
        ":melting_face:"*5,
        ":mushroom_cloud:"*6
    ]

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


def format_slack_table(items: List[dict]) -> str:
    """
    Given a list of dicts with keys: Product, Version, Severity, CVE,
    return a string containing a code-block table with evenly padded columns.

    Inputs
    ======
    items: List[dict]
        A list of dicts with keys Product, Version, Severity, CVE

    Returns
    =======
    str: A string containing a code-block table with evenly padded columns
    """
    if not items:
        return "```Nothing to see here...```"

    # 1) Define your columns and compute the max width for each
    headers = ["Product", "Version", "Severity", "CVE"]
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


def cbt_message(cbt: dict) -> str:
    """
    Convert list of cve-bin-tool CVEs to a table for Slack.

    Inputs
    ======
    cbt: dict
        Dictionary of cve-bin-tool CVEs

    Returns
    =======
    str: string with table of CVEs
    """

    if not cbt:
        return ""

    key_map = {
        "memgraph": "\nMemgraph Binaries\n------------------\n",
        "lang": "\nLanguage Packages\n------------------\n",
        "apt": "\nAPT Packages\n------------------\n"
    }

    msg = "CVE-bin-tool Summary:\n"
    msg += "============================"

    total_items = 0

    for key, val in cbt.items():
        
        items = []
        for item in val["cve"]:
            if item["severity"] in ["CRITICAL"]:  # only showing critical for now
                items.append({
                    "Product": item["product"],
                    "Version": item["version"],
                    "Severity": item["severity"],
                    "CVE": item["id"],
                })

        if len(items) > 0:
            msg += key_map[key]
            total_items += len(items)
            table = format_slack_table(items)
            msg += f"\n{table}\n"
    
    if total_items == 0:
        return ""

    return msg


def grype_trivy_message(cves: dict, name: str) -> str:
    """
    Convert list of Grype CVEs to a part of the Slack message.

    Inputs
    ======
    cves: dict
        Grype CVEs to be converted.
    name: str
        Name of the tool that generated the CVEs.

    Returns
    =======
    str: string contaiing table of CVEs
    """

    if not cves:
        return ""

    msg = f"{name} Summary:\n"
    msg += "============================"
    items = []
    for item in cves["cve"]:
        if item["severity"].upper() in ["CRITICAL"]:
            items.append({
                "Product": item["product"],
                "Version": item["version"],
                "Severity": item["severity"],
                "CVE": item["id"]
            })

    if len(items) == 0:
        return ""

    if name == "Trivy":
        cve_severity_score = {
            "CRITICAL": 4,
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1,
            "NEGLIGIBLE": 0,
            "UNKNOWN": -1,
        }

        # trivy picks up lots of CVEs for linux-libc-dev, so let's summarize
        libc_cves = [item for item in items if item["Product"] == "linux-libc-dev"]

        if len(libc_cves) > 10:
            items = [item for item in items if item["Product"] != "linux-libc-dev"]
            version = list(set([item["Version"] for item in libc_cves]))
            if len(version) == 1:
                version = version[0]
            else:
                version = "Multiple"

            scores = [cve_severity_score[item["Severity"]] for item in libc_cves]
            severity = {v: k for k, v in cve_severity_score.items()}.get(max(scores), "????")

            items.append({
                "Product": "linux-libc-dev",
                "Version": version,
                "Severity": severity,
                "CVE": "Too Many CVEs",
            })

    table = format_slack_table(items)
    return msg + f"\n{table}\n"


def create_slack_message(arch: str, image_type: str, cbt: dict, grype: dict, trivy: dict) -> str:
    """
    Formats the Slack message to be sent.

    Inputs
    ======
    arch: str
        The architecture of the image to be scanned.
    image_type: str, choices=["memgraph", "mage"]
        The type of image to be scanned.
    cbt: dict
        The CVEs from cve-bin-tool.
    grype: dict, optional
        The CVEs from Grype. Defaults to None.
    trivy: dict, optional
        The CVEs from Trivy. Defaults to None.

    Returns
    =======
    str: The formatted Slack message.
    """

    arch_str = "x86_64" if arch == "amd64" else "aarch64"
    name = "Memgraph" if image_type == "memgraph" else "MAGE"

    msg_start = f"Vulnerability Scan Results for *{name}* (Docker {arch_str})...\n\n"

    summary_list = [
        v["summary"] for _, v in cbt.items()
    ] + [
        grype["summary"] if grype else {},
        trivy["summary"] if trivy else {}
    ]

    # combine summaries
    _, summary_msg = combine_summaries(summary_list)

    cbt_msg = cbt_message(cbt)
    grype_msg = grype_trivy_message(grype, "Grype")
    trivy_msg = grype_trivy_message(trivy, "Trivy")

    msg = "\n".join([msg_start, summary_msg, grype_msg, trivy_msg, cbt_msg])

    return msg


def post_message(msg: str) -> None:
    """
    Post message to Slack webhook.

    Inputs
    ======
    msg: str
        Message containing vulnerability summary.
    """

    url = os.getenv("INFRA_WEBHOOK_URL")
    try:
        response = requests.post(
            url,
            json={"text": msg}
        )
    except Exception:
        print(f"Response: {response.status_code}")


def save_full_vulnerability_list(summary_list: List[list]) -> None:
    """
    Save full vulnerability list in a file for further processing.

    Inputs
    ======
    summary_list: List[list]
        A list containing vulnerability information for each CVE.
    """

    cves = []
    for item in summary_list:
        cves.extend(item)

    table = format_slack_table(cves)

    with open(f"{CVE_DIR}/full-cve-list.txt", "w") as f:
        f.write(table)


def main(arch: str, image_type: str) -> None:
    """
    Collect vulnerability results and send a Slack message.

    Inputs
    ======
    arch: str
        The architecture of the image to be scanned.
    image_type: str
        The type of image to be scanned.

    """

    # collect results
    cbt = parse_cbt_summary()
    grype = parse_grype_summary()
    trivy = parse_trivy_summary()

    msg = create_slack_message(arch, image_type, cbt, grype, trivy)
    post_message(msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("arch", type=str, default="amd64")
    parser.add_argument('image_type', type=str, choices=['memgraph', 'mage'], default='mage')
    args = parser.parse_args()

    main(args.arch, args.image_type)
