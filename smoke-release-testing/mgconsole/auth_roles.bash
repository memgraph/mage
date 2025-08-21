#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_show_database_settings() {
  # TODO(gitbuda): args became useless
  __host="$1"
  __port="$2"
  echo "SHOW DATABASE SETTINGS;" | $__mgconsole_admin
}

test_auth_roles() {
  # TODO(gitbuda): args became useless
  __host="$1"
  __port="$2"
  echo "FEATURE: Auth Roles"

  echo "CREATE ROLE IF NOT EXISTS test_reader;" | $__mgconsole_admin
  echo "SHOW ROLES;" | $__mgconsole_admin
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.

  # NOTE: If you want to run custom memgraph binary just set MEMGRAPH_BUILD_PATH to your memgraph build directory.
  run_memgraph_binary "--bolt-port $MEMGRAPH_DEFAULT_PORT --log-level=TRACE --log-file=mg_test_auth_roles_enterprise.logs --data-directory=test_auth_roles_enterprise"
  wait_for_memgraph $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_DEFAULT_PORT
  # test_show_database_settings $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_DEFAULT_PORT
  test_auth_roles $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_DEFAULT_PORT
  cleanup_memgraph_binary_processes

  rm -rf $MG_BUILD_PATH/test_auth_roles_enterprise/auth
  ORIG_MEMGRAPH_ENTEPRISE_LICENSE="$MEMGRAPH_ENTERPRISE_LICENSE"
  ORIG_MEMGRAPH_ORGANIZATION_NAME="$MEMGRAPH_ORGANIZATION_NAME"
  unset MEMGRAPH_ENTERPRISE_LICENSE
  unset MEMGRAPH_ORGANIZATION_NAME
  # NOTE: License is not stored in settings.
  run_memgraph_binary "--bolt-port 7688 --log-level=TRACE --log-file=mg_test_auth_roles_community.logs --data-directory=test_auth_roles_enterprise"
  wait_for_memgraph $MEMGRAPH_DEFAULT_HOST 7688
  # test_show_database_settings $MEMGRAPH_DEFAULT_HOST 7688
  test_auth_roles $MEMGRAPH_DEFAULT_HOST 7688
  export MEMGRAPH_ENTERPRISE_LICENSE="$ORIG_MEMGRAPH_ENTEPRISE_LICENSE"
  export MEMGRAPH_ORGANIZATION_NAME="$ORIG_MEMGRAPH_ORGANIZATION_NAME"
fi
