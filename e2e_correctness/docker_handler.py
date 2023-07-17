import subprocess


def start_memgraph_mage_container(image_name:str, container_name:str, port:int) -> str:
    """
    docker run \
    --name memgraph_mage \
    -p 7687:7687 -p 7444:7444   memgraph-mage:test --also-log-to-stderr --log-level=TRACE --telemetry-enabled=False
    """
    command_start_memgraph = f"docker run \
                        --rm \
                        --name {container_name} \
                        -p 7687:7687 -p 7444:7444   \
                        -d \
                        {image_name} --also-log-to-stderr --log-level=TRACE --telemetry-enabled=False"
    memgraph_started = subprocess.run(command_start_memgraph, shell=True, stdout=subprocess.PIPE)
    memgraph_started.check_returncode()
    return memgraph_started.stdout.decode("utf-8")


def start_neo4j_apoc_container(image_name:str, container_name:str, port:int) -> str:
    """
    docker run \
    --name neo4j_test_driven \
    -p 7474:7474 -p 7688:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=none  
    -e NEO4J_apoc_export_file_enabled=true \ 
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
    neo4j:latest
    """

    command_start_neo = f"docker run --rm \
                        --name {container_name}  \
                        -p 7474:7474 -p {port}:7687 \
                        -d -v $HOME/neo4j/data:/data \
                        -v $HOME/neo4j/logs:/logs \
                        -v $HOME/neo4j/import:/var/lib/neo4j/import \
                        -v $HOME/neo4j/plugins:/plugins \
                        --env NEO4J_AUTH=none  \
                        -e NEO4J_apoc_export_file_enabled=true \
                        -e NEO4J_apoc_import_file_enabled=true \
                        -e NEO4J_apoc_import_file_use__neo4j__config=true  \
                        -e NEO4JLABS_PLUGINS=\[\"apoc\"\]  {image_name}"
    neo_started = subprocess.run(command_start_neo, shell=True, stdout=subprocess.PIPE)
    neo_started.check_returncode()
    return neo_started.stdout.decode("utf-8")


def stop_container(container_id:str)->bool:

    command_stop_container = f"docker stop {container_id}"
    docker_stoped = subprocess.run(command_stop_container, shell=True, stdout=subprocess.PIPE)
    try:
        docker_stoped.check_returncode()
    except Exception as e:
        print(e)
        return False
    return True

def remove_container(container_id:str)->bool:

    command_remove_container = f"docker rm {container_id}"
    docker_stoped = subprocess.run(command_remove_container, shell=True, stdout=subprocess.PIPE)
    try:
        docker_stoped.check_returncode()
    except Exception as e:
        print(e)
        return False
    return True

def main():
    
    neo4j_container_id = start_neo4j_apoc_container("neo4j:latest", "neo4jtest", 7688)

    assert stop_container(container_id=neo4j_container_id) and remove_container(container_id=neo4j_container_id), f"Container \"{neo4j_container_id}\" not stoped or removed, check for errors"



if __name__ == "__main__":
    main()

    