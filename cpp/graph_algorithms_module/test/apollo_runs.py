#!/usr/bin/env python3
import json
import os
import re
import subprocess

# paths
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
WORKSPACE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
TESTS_DIR_REL = os.path.join("..", "build_release", "test")
TESTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, TESTS_DIR_REL))

# list of long running tests
LARGE_TESTS = ["unit__algorithms_cutsets"]

# ctest tests
ctest_output = subprocess.run(["ctest", "-N"], cwd=TESTS_DIR, check=True,
        stdout=subprocess.PIPE).stdout.decode("utf-8")
tests = []

# gather all ctest tests
CTEST_DELIMITER = "__"
for row in ctest_output.split("\n"):
    # Filter rows only containing tests.
    if not re.match("^\s*Test\s+#", row): continue
    name = row.split(":")[1].strip()
    path = os.path.join(TESTS_DIR_REL, name.replace(CTEST_DELIMITER, "/", 1))
    tests.append((name, path))

tests.sort()

runs_unit = []
runs_coverage = []
for test in tests:
    name, path = test
    dirname, basename = os.path.split(path)
    files = [basename]

    # skip all tests that aren't unit
    if not name.startswith("unit"):
        continue

    # larger timeout for large tests
    prefix = ""
    if name in LARGE_TESTS:
        prefix = "TIMEOUT=600 "

    runs_unit.append({
        "name": name,
        "cd": dirname,
        "commands": prefix + "./" + basename,
        "infiles": files,
        "outfile_paths": [],
    })

    dirname = dirname.replace("/build_release/", "/build_coverage/")
    curdir_abs = os.path.normpath(os.path.join(SCRIPT_DIR, dirname))
    curdir_rel = os.path.relpath(curdir_abs, WORKSPACE_DIR)
    outfile_paths = ["\./" + curdir_rel.replace(".", "\\.") + "/default\\.profraw"]

    runs_coverage.append({
        "name": "coverage" + name[4:],
        "cd": dirname,
        "commands": prefix + "./" + basename,
        "infiles": files,
        "outfile_paths": outfile_paths,
    })

runs = runs_unit + runs_coverage
print(json.dumps(runs, indent=4, sort_keys=True))
