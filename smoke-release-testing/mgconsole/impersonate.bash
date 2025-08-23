#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_impersonate_user() {
  echo "FEATURE: Impersonate User"
  echo "GRANT IMPERSONATE_USER * TO admin;" | $MGCONSOLE_NEXT_ADMIN
}
