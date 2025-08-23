#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_show_database_settings() {
  echo "FEATURE: Show Database Settings"
  echo "SHOW DATABASE SETTINGS;" | $MGCONSOLE_NEXT_ADMIN
}

test_auth_roles() {
  echo "FEATURE: Auth Roles"
  echo "CREATE ROLE IF NOT EXISTS test_reader;" | $MGCONSOLE_NEXT_ADMIN
  echo "SHOW ROLES;" | $MGCONSOLE_NEXT_ADMIN
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  echo "pass"
fi
