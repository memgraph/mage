import subprocess


def start_memgraph_mage_container(image_name:str, container_name:str, port:int) -> str:
    """
    Starts Memgraph MAGE container in detached mode with following options set:
        --also-log-to-stderr 
        --log-level=TRACE 
        --telemetry-enabled=False
    
    This way in case something breaks we can use logs from Memgraph.
    """
    command_start_memgraph = f"docker run \
                        --rm \
                        --name {container_name} \
                        -p {port}:7687 -p 7444:7444   \
                        -d \
                        {image_name} \
                        --also-log-to-stderr \
                        --log-level=TRACE \
                        --telemetry-enabled=False"
    memgraph_started = subprocess.run(command_start_memgraph, shell=True, stdout=subprocess.PIPE)
    memgraph_started.check_returncode()
    return memgraph_started.stdout.decode("utf-8")


def start_neo4j_apoc_container(image_name:str, container_name:str, port:int) -> str:
    """
    Starts Neo4j container without any volumens on port from parameters.

    Following flags need to be set in order to use import and export:
        -e NEO4J_apoc_export_file_enabled=true \
        -e NEO4J_apoc_import_file_enabled=true \
        -e NEO4J_apoc_import_file_use__neo4j__config=true  \
    Uses latest neo4j with apoc plugin included

    Uses --rm on containers, so they don't need to be manually removed.
    """
    command_start_neo = f"docker run --rm \
                        --name {container_name}  \
                        -p 7474:7474 -p {port}:7687 \
                        --rm \
                        -d \
                        -v $HOME/neo4j/plugins:/plugins \
                        --env NEO4J_AUTH=none  \
                        -e NEO4J_apoc_export_file_enabled=true \
                        -e NEO4J_apoc_import_file_enabled=true \
                        -e NEO4J_apoc_import_file_use__neo4j__config=true  \
                        -e NEO4JLABS_PLUGINS=\[\"apoc\"\]  {image_name}"
    neo_started = subprocess.run(command_start_neo, shell=True, stdout=subprocess.PIPE)
    neo_started.check_returncode()
    return neo_started.stdout.decode("utf-8")


def stop_container(container_id:str) -> bool:
    command_stop_container = f"docker stop {container_id}"
    docker_stoped = subprocess.run(command_stop_container, shell=True, stdout=subprocess.PIPE)
    try:
        docker_stoped.check_returncode()
    except Exception as e:
        print(e)
        return False
    return True
    