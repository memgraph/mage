import os
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import hashlib
import random
import time


def file_hash(output_dir):

    hash = hashlib.sha256(
        str(random.random()).encode("utf-8")
    ).hexdigest()

    return f"{output_dir}/{hash}.json"


def find_memgraph_files(start_dir):
    """

    """
    matches = []
    for dirpath, _, filenames in os.walk(start_dir):
        for filename in filenames:
            fullpath = f"{dirpath}/{filename}"
            try:
                st = os.lstat(fullpath)
            except OSError:
                # Skip files we can’t stat
                continue

            is_symlink = stat.S_ISLNK(st.st_mode)

            # Only consider regular files; skip symlinks
            if is_symlink:
                continue

            matches.append(fullpath)
    return matches


def run_cve_scan(directory, output_dir):
    """
    Run cve-bin-tool on a single directory with JSON output to a file,
    using '-u never' and '-f json'. Returns the JSON string read from the file.
    Captures stdout/stderr so as not to interfere with the progress bar.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct a safe filename from the directory path
    # e.g., "/usr/local/bin" → "usr_local_bin.json"
    output_path = file_hash(output_dir)

    cmd = [
        "cve-bin-tool",
        "-u", "never",       # Never update the local CVE database
        "-f", "json",        # Output format: JSON
        "-o", output_path,   # Write JSON results to this file
        directory            # Directory to scan
    ]

    t0 = time.time()
    # Run the command, capturing stdout/stderr
    _ = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    t1 = time.time()

    # Read and return the JSON contents from the output file
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except OSError as e:
        raise RuntimeError(f"Could not read JSON output for {directory!r}: {e}")

    return data, t1-t0


def scan_directories_with_progress(dirs_to_scan, output_dir="tmp", max_workers=20):
    """
    Given a list of directories, scan each one in parallel using cve-bin-tool.
    - Uses '-u never' and '-f json'.
    - Writes each JSON result into 'output_dir'.
    - Shows a tqdm progress bar that advances as each scan completes.
    - Returns a dict mapping directory → (json_str or None, output_file_path).
      If a scan fails, json_str will be None, and the exception is printed.
    """
    # Prepare the results dictionary
    results = []
    times = []
    outdir = []
    # Submit one task per directory
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dir = {
            executor.submit(run_cve_scan, d, output_dir): d
            for d in dirs_to_scan
        }

        # Wrap as_completed in tqdm for a live progress bar
        for future in tqdm(
            as_completed(future_to_dir),
            total=len(future_to_dir),
            desc="Scanning directories",
            unit="dir"
        ):
            directory = future_to_dir[future]
            try:
                json_data, dt = future.result()
                if isinstance(json_data, list):
                    results.extend(json_data)
                else:
                    results.append(json_data)
                times.append(dt)
                outdir.append(directory)
            except Exception as exc:
                print(f"Error scanning {directory!r}: {exc}")

    with open("cve-bin-tool-memgraph-summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, times, outdir


def place_slowest_first(rootfs, directories):

    # these directories are the slowest to scan, so to speed things up a tiny bit,
    # let's scan them first so they are being done while other threads deal with
    # the quick ones (assumes that we have more threads than slow ones!)

    slow_dirs = [
        "usr/lib/memgraph/memgraph",
        "usr/bin/mg_import_csv",
        "usr/bin/mg_dump",
        "usr/bin/mgconsole",
    ]

    outdirs = []
    for sd in slow_dirs:
        slow_dir = f"{rootfs}/{sd}"
        if os.path.isdir(slow_dir) or os.path.isfile(slow_dir):
            outdirs.append(slow_dir)
        if slow_dir in directories:
            directories.remove(slow_dir)

    outdirs = outdirs + directories
    return outdirs


if __name__ == "__main__":
    # Example usage:

    files = find_memgraph_files("rootfs/usr/lib/memgraph")
    files = place_slowest_first("rootfs", files)
    results, times, outdir = scan_directories_with_progress(files)
