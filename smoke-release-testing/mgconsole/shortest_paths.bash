#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_shortest_paths() {
  __host="$1"
  __port="$2"
  echo "FEATURE: KShortest paths and AllShortest paths"
  echo "MATCH (n) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  
  create_graph_query="
    CREATE
        (a:Node {id: 'A'}),
        (b:Node {id: 'B'}),
        (c:Node {id: 'C'}),
        (d:Node {id: 'D'}),
        (e:Node {id: 'E'}),
        (f:Node {id: 'F'}),
        (g:Node {id: 'G'}),
        (a)-[:REL {weight: 1}]->(b),
        (a)-[:REL {weight: 2}]->(c),
        (b)-[:REL {weight: 1}]->(e),
        (c)-[:REL {weight: 1}]->(e),
        (a)-[:REL {weight: 3}]->(e),
        (b)-[:REL {weight: 2}]->(d),
        (d)-[:REL {weight: 1}]->(e),
        (a)-[:REL {weight: 1}]->(f),
        (f)-[:REL {weight: 1}]->(g),
        (g)-[:REL {weight: 1}]->(e);
    "
  echo "$create_graph_query" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port

  echo "SUBFEATURE: Test KShortest paths - find top 3 shortest paths from A to E"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*KShortest]->(n2) RETURN p;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test KShortest paths with path length bounds"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*KShortest 2..4]->(n2) RETURN p;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test KShortest paths with limit"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*KShortest 2..4 | 5]->(n2) RETURN p;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test KShortest paths with hops limit"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*KShortest 1..3]->(n2) RETURN p;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test KShortest paths with node filtering"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*KShortest]->(n2) WHERE ALL(n IN nodes(p) WHERE n.id IN ['A', 'B', 'C', 'E']) RETURN p;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test KShortest paths with edge filtering"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*KShortest]->(n2) WHERE ALL(r IN relationships(p) WHERE r.weight <= 2) RETURN p;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test KShortest paths with weighted edges (using weight property)"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*KShortest]->(n2) RETURN p, reduce(total = 0, r IN relationships(p) | total + r.weight) AS total_weight ORDER BY total_weight;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test KShortest paths with result ordering"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*KShortest]->(n2) RETURN p, length(p) AS path_length ORDER BY path_length;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test KShortest paths with null start/end nodes (should return empty set)"
  echo "MATCH (n1:Node {id: 'X'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*KShortest]->(n2) RETURN p;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test KShortest paths with same start and end node (should return empty set)"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'A'}) WITH n1, n2 MATCH p=(n1)-[*KShortest]->(n2) RETURN p;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test KShortest paths with the LIMIT clause"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*KShortest]->(n2) RETURN p LIMIT 1000;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
 
  echo "SUBFEATURE: Test AllShortest paths - find all shortest paths from A to E (with weight lambda)"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*AllShortest (r, n | r.weight)]->(n2) RETURN p;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test AllShortest paths with the upper bound (with weight lambda)"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*AllShortest ..4 (r, n | r.weight)]->(n2) RETURN p;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SUBFEATURE: Test AllShortest paths getting the path length (with weight lambda)"
  echo "MATCH (n1:Node {id: 'A'}), (n2:Node {id: 'E'}) WITH n1, n2 MATCH p=(n1)-[*AllShortest (r, n | r.weight)]->(n2) RETURN DISTINCT length(p) AS path_length ORDER BY path_length;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port

  echo "MATCH (n) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "KShortest and AllShortest paths testing completed successfully"
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_test_shortest_paths.logs" test_shortest_paths
fi
