import itertools
from os import wait
import time

from datasets import load_dataset
import docker
import mgclient


# Under Mac -> Settings > Advanced > Allow the default Docker socket to be used
# (requires password).
DOCKER_CLIENT = docker.from_env()


def container_exists(name):
    try:
        DOCKER_CLIENT.containers.get(name)
        return True
    except docker.errors.NotFound:
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def start_memgraph():
    if container_exists("test_text2cypher_queries"):
        return
    DOCKER_CLIENT.containers.run(
        "memgraph/memgraph",
        detach=True,
        auto_remove=True,
        name="test_text2cypher_queries",
        ports={"7687": 7687},
        command=["--telemetry-enabled=False"],
    )


def execute_query(query):
    try:
        conn = mgclient.connect(host="127.0.0.1", port=7687)
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(query)
        cursor.fetchall()
        cursor.close()
        conn.close()
        return True
    except:
        return False


def is_memgraph_alive():
    return execute_query("RETURN 1;")


def wait_memgraph():
    while True:
        if is_memgraph_alive():
            break
        time.sleep(0.1)


def start_and_wait_memgraph():
    start_memgraph()
    # NOTE: There is some weird edgecase in deleting/starting...
    time.sleep(1)
    start_memgraph()
    wait_memgraph()


dataset_path = "neo4j/text2cypher-2025v1"
dataset = load_dataset(dataset_path)
all_items_iter = itertools.chain(dataset["train"], dataset["test"])

start_and_wait_memgraph()
tried_queres = 0
passed_queries = 0
failed_queries = 0
number_of_restarts = 0
for item in all_items_iter:
    if not is_memgraph_alive():
        number_of_restarts += 1
        start_and_wait_memgraph()
    query = item["cypher"].replace("\\n", " ")
    tried_queres += 1
    print(f"Executing: {query}")
    if execute_query(query):
        passed_queries += 1
    else:
        failed_queries += 1
print(f"The number of tried queries: {tried_queres}")
print(f"The number of passed queries: {passed_queries}")
print(f"The number of failed queries: {failed_queries}")
print(f"The number of memgraph restarts: {number_of_restarts}")

# TODO(gitbuda): Implement a generator based on the schema and execute queries
# on top of the real data.
