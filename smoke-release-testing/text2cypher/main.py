import itertools
from os import wait
import time
import logging
from datetime import datetime

from datasets import load_dataset
import docker
import mgclient

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"text2cypher_test_{timestamp}.log"

# Create a formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Configure file handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

# Configure console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Under Mac -> Settings > Advanced > Allow the default Docker socket to be used
# (requires password).
DOCKER_CLIENT = docker.from_env()


def container_exists(name):
    try:
        DOCKER_CLIENT.containers.get(name)
        logger.debug(f"Container {name} already exists")
        return True
    except docker.errors.NotFound:
        logger.debug(f"Container {name} does not exist")
        return False
    except Exception as e:
        logger.debug(f"An error occurred while checking container {name}: {e}")
        return False


def start_memgraph():
    if container_exists("test_text2cypher_queries"):
        return
    logger.info("Starting Memgraph container...")
    try:
        DOCKER_CLIENT.containers.run(
            "memgraph/memgraph:3.3.0",
            detach=True,
            auto_remove=True,
            name="test_text2cypher_queries",
            ports={"7687": 7687},
            command=["--telemetry-enabled=False"],
        )
        logger.info("Memgraph container started successfully")
    except Exception as e:
        logger.error(f"Failed to start Memgraph container: {e}")
        raise


# TODO(gitbuda): Add a timeout to the query execution.
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
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        return False


def is_memgraph_alive(reason=None):
    # NOTE: This is tricky because sometimes the client error is "Can't assign
    # requested address" while memgraph is still alive.
    if reason:
        logger.debug(f"Checking if Memgraph is alive... ({reason})")
    else:
        logger.debug("Checking if Memgraph is alive...")
    if not container_exists("test_text2cypher_queries"):
        return False
    probe_query = "RETURN 1;"
    count = 20
    while True:
        if execute_query(probe_query):
            return True
        time.sleep(0.1)
        count -= 1
        if count == 0:
            logger.error(
                "I couldn't get back from memgraph for a long time, exiting..."
            )
            return False


def wait_memgraph():
    logger.info("Waiting for Memgraph to become available...")
    while True:
        if is_memgraph_alive():
            logger.info("Memgraph is now available")
            break
        time.sleep(0.1)


def start_and_wait_memgraph():
    start_memgraph()
    # NOTE: There is some weird edgecase in deleting/starting...
    time.sleep(1)
    start_memgraph()
    wait_memgraph()


logger.info("Loading dataset...")
dataset_path = "neo4j/text2cypher-2025v1"
dataset = load_dataset(dataset_path)
all_items_iter = itertools.chain(dataset["train"], dataset["test"])
logger.info("Dataset loaded")

start_and_wait_memgraph()
tried_queres = 0
passed_queries = 0
failed_queries = 0
number_of_restarts = 0
queries_crasing_memgraph_file = "queries_crashing_memgraph_file.cypher"

with open(queries_crasing_memgraph_file, "w") as f:
    for item in all_items_iter:
        if not is_memgraph_alive("did previous query crash memgraph?"):
            logger.debug("The previous query crashed memgraph, restarting...")
            number_of_restarts += 1
            start_and_wait_memgraph()
        else:
            logger.debug("All good, Memgraph is still alive")
        query = item["cypher"].replace("\\n", " ")
        tried_queres += 1
        logger.info(f"Executing: {query}")
        # TODO(gitbuda): Some queries have params -> pass relevant params somehow.
        if execute_query(query):
            passed_queries += 1
        else:
            failed_queries += 1
            if not is_memgraph_alive():
                f.write(f"{query};\n")

logger.info("Test Results Summary:")
logger.info(f"The number of tried queries: {tried_queres}")
logger.info(f"The number of passed queries: {passed_queries}")
logger.info(f"The number of failed queries: {failed_queries}")
logger.info(f"The number of memgraph restarts: {number_of_restarts}")
logger.info(f"Failed queries have been written to: {queries_crasing_memgraph_file}")
logger.info(f"Full log has been written to: {log_file}")

# TODO(gitbuda): Implement a generator based on the schema and execute queries
# on top of the real data.
