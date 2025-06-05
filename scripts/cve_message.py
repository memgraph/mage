import requests
import os
from typing import List, Tuple
import json
import argparse


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


def severity_summary(data: List[dict]) -> dict:

    summary = {}
    for cve in data:
        severity = cve["severity"]
        if severity not in summary:
            summary[severity] = 0
        summary[severity] += 1

    return summary


def reformat_cbt_data(data):

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
    This will read in the `cve-bin-tool-summary.txt` and scan for detected
    vulnerabilities.
    """

    parts = ["memgraph", "lang", "apt"]
    out = {}
    for part in parts:
        fname = f"cve-bin-tool-{part}-summary.json"
        print(fname)
        if os.path.isfile(fname):
            print("here")
            data = read_summary_file(fname)
            data = reformat_cbt_data(data)
            out[part] = {
                "summary": severity_summary(data),
                "cve": data
            }

    return out


def reformat_grype_data(data):

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
    This will read in the `grype-summary.txt`
    """

    data = read_summary_file("grype-summary.json")
    if not data:
        return {}

    results = data["matches"]
    cves = reformat_grype_data(results)
    summary = severity_summary(cves)

    return {
        "summary": summary,
        "cve": cves
    }


def reformat_trivy_data(data):

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

    data = read_summary_file("trivy-summary.json")
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

    summary = {}
    for dct in summary_list:
        dct = {k: v for k, v in dct.items() if v > 0}
        print(dct)
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


def format_slack_table(items):
    """
    Given a list of dicts with keys: Package, Version, Severity, CVE,
    return a string containing a code-block table with evenly padded columns.
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


def cbt_message(cbt):

    if not cbt:
        return ""

    key_map = {
        "memgraph": "\nMemgraph Binaries\n------------------\n",
        "lang": "\nLanguage Packages\n------------------\n",
        "apt": "\nAPT Packages\n------------------\n"
    }

    msg = "CVE-bin-tool Summary:\n"
    msg += "============================"

    for key, val in cbt.items():
        msg += key_map[key]
        items = []
        for item in val["cve"]:
            if item["severity"] in ["HIGH", "CRITICAL"]:  # should we show everything?
                items.append({
                    "Product": item["product"],
                    "Version": item["version"],
                    "Severity": item["severity"],
                    "CVE": item["id"],
                })

        table = format_slack_table(items)
        msg += f"\n{table}\n"
    return msg


def grype_trivy_message(cves, name):

    if not cves:
        return ""

    msg = f"{name} Summary:\n"
    msg += "============================"
    items = []
    for item in cves["cve"]:
        if item["severity"].upper() in ["HIGH", "CRITICAL"]:
            items.append({
                "Product": item["product"],
                "Version": item["version"],
                "Severity": item["severity"],
                "CVE": item["id"]
            })

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
        print(len(libc_cves))
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


def create_slack_message(arch, image_type, cbt, grype, trivy) -> str:

    arch_str = "x86_64" if arch == "amd" else "aarch64"
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


def post_message(msg):

    url = os.getenv("INFRA_WEBHOOK_URL")
    try:
        response = requests.post(
            url,
            json={"text": msg}
        )
    except Exception:
        print(f"Response: {response.status_code}")


def save_full_vulnerability_list(summary_list):

    cves = []
    for item in summary_list:
        cves.extend(item)

    table = format_slack_table(cves)

    with open("full-cve-list.txt", "w") as f:
        f.write(table)


def main(arch, image_type):

    # collect results
    cbt = parse_cbt_summary()
    grype = parse_grype_summary()
    trivy = parse_trivy_summary()

    msg = create_slack_message(arch, image_type, cbt, grype, trivy)
    post_message(msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("arch", type=str, default="amd")
    parser.add_argument('image_type', type=str, choices=['memgraph', 'mage'], default='mage')
    args = parser.parse_args()

    main(args.arch, args.image_type)
