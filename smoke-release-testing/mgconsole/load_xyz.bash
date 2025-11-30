#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_load_xyz() {
  echo "FEATURE: LOAD CSV/JSONL/PARQUET"
  # Added in v3.7.
  # TODO(gitbuda): Create test files and mount them when starting containers.
  run_next "LOAD PARQUET FROM '/data/nodes.parquet' AS row CREATE (n:Node {id: row.id});"
  run_next "MATCH (n) RETURN n;"
}
