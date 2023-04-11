# map current directory to /mage/ in the container
docker run -d --rm -v $(pwd):/mage/ -v /usr/lib/memgraph/query_modules --name mage memgraph/memgraph-mage:1.7.0-dev
curl https://raw.githubusercontent.com/memgraph/memgraph/master/include/_mgp_mock.py --output python/_mgp_mock.py;
curl https://raw.githubusercontent.com/memgraph/memgraph/master/include/mgp_mock.py --output python/mgp_mock.py
#docker cp . mage:/mage/
docker exec -u root -it mage /bin/bash
docker run -it --rm -p 7687:7687 -p 7444:7444 -p 3000:3000 -v mg_lib:/var/lib/memgraph --name memgraph memgraph/memgraph-platform
docker cp mage:/usr/lib/memgraph/query_modules .
#docker cp query_modules/ memgraph:/usr/lib/memgraph
docker cp query_modules/particle_filtering.so memgraph:/usr/lib/memgraph/query_modules/

#docker cp mage:/usr/lib/memgraph/query_modules .; docker cp query_modules/particle_filtering.so memgraph:/usr/lib/memgraph/query_modules/

#python3 setup build -p /usr/lib/memgraph/query_modules