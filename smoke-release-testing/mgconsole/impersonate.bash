#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_impersonate_user() {
  # TODO(gitbuda): args became useless
  __host="$1"
  __port="$2"
  echo "FEATURE: Impersonate User"

  echo "GRANT IMPERSONATE_USER * TO admin;" | $__mgconsole_admin
}
