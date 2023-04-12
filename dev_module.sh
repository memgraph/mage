# script to facilitate development of query modules. It runs mage dev container (for building custom query modules)
# and memgraph-platform container (for testing out the query modules against a memgraph database instance).
# after running this script you can use the 'update' and 'check' functions to update query modules and run cargo check
#curl https://raw.githubusercontent.com/memgraph/memgraph/master/include/_mgp_mock.py --output python/_mgp_mock.py;
#curl https://raw.githubusercontent.com/memgraph/memgraph/master/include/mgp_mock.py --output python/mgp_mock.py

# run mage dev container, mapping current directory to /mage/ in container
docker run -d --rm -v $(pwd):/mage/ --name mage memgraph/memgraph-mage:1.7.0-dev
# run memgraph-platfom container
docker run -d -it --rm -p 7687:7687 -p 7444:7444 -p 3000:3000 --name memgraph memgraph/memgraph-platform

# function to run cargo check in mage dev container
function check() {
    docker exec -u root -it mage /bin/bash -c 'source ~/.bashrc; cargo check --manifest-path rust/particle-filtering/Cargo.toml';
}

# function to update query modules in memgraph container
function update() {
    docker exec -u root -it mage /bin/bash -c "source ~/.bashrc ; python3 setup build -p /usr/lib/memgraph/query_modules";
    docker cp mage:/usr/lib/memgraph/query_modules .;
    docker cp query_modules/particle_filtering.so memgraph:/usr/lib/memgraph/query_modules/;
    docker exec -it memgraph bash -c 'echo "CALL mg.load_all();" | mgconsole';
}

update

