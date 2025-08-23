#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_composite_indices() {
  echo "FEATURE: Label Property Composit Index"

  echo "CREATE INDEX ON :Label(prop1, prop2);" | $MGCONSOLE_NEXT_DEFAULT
  echo "CREATE (:Label {prop1:0, prop2: 1});" | $MGCONSOLE_NEXT_DEFAULT
  echo "EXPLAIN MATCH (n:Label {prop1:0, prop2: 1}) RETURN n;" | $MGCONSOLE_NEXT_DEFAULT | grep -E "ScanAllByLabelProperties \(n :Label \{prop1, prop2\}\)"

  echo "SHOW INDEXES;" | $MGCONSOLE_NEXT_DEFAULT
}

test_nested_indices() {
  echo "FEATURE: Nested Indices"

  echo "CREATE INDEX ON :Project(delivery.status.due_date);" | $MGCONSOLE_NEXT_DEFAULT
  echo "CREATE (:Project {delivery: {status: {due_date: date('2025-06-04'), milestone: 'v3.14'}}});" | $MGCONSOLE_NEXT_DEFAULT
  echo "EXPLAIN MATCH (proj:Project) WHERE proj.delivery.status.due_date = date('2025-06-04') RETURN *;" | $MGCONSOLE_NEXT_DEFAULT | grep -E "ScanAllByLabelProperties"

  echo "SHOW INDEXES;" | $MGCONSOLE_NEXT_DEFAULT
}
