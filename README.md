# MAGE - Memgraph Advanced Graph Extensions :crystal_ball:

This open-source repository contains all available query modules written by the team behind Memgraph and its users. You can find and contribute implementations of various algorithms in multiple programming languages, all runnable inside Memgraph. This project aims to give everyone the tools they need to tackle the most challenging graph problems. 

## MAGE Query modules

Memgraph introduces the concept of **query modules**, user-defined procedures that extend the Cypher query language. These procedures are grouped into modules that can be loaded into Memgraph. You can find more information on query modules in the official [documentation](https://docs.memgraph.com/memgraph/database-functionalities/query-modules/built-in-query-modules).

Query modules implemented in Python:
* [nxalg](python/nxalg.py): A module that provides NetworkX integration with Memgraph and implements many NetworkX algorithms.  
* [graph_analyzer.py](python/graph_analyzer.py): This Graph Analyzer query module offers insights about the stored graph or a subgraph.
* [distance_calculator.py](python/distance_calculator.py): Module for finding the geographical distance between two points defined with 'lng' and 'lat' coordinates.
* [tsp.py](python/tsp.py): An algorithm for finding the shortest possible route that visits each vertex exactly once.
* [set_cover.py](python/set_cover.py): The algorithm for finding minimum cost subcollection of sets that covers all elements of a universe.
* [collapse.py](python/collapse.py): Module for collapsing specifically connected graph nodes into different subgraphs.

Query modules implemented in C/C++:
* [connectivity_module.cpp](cpp/connectivity_module/connectivity_module.cpp): A module that finds weakly connected components in a graph.
* [biconnected_components_module.cpp](cpp/biconnected_components_module/biconnected_components_module.cpp): Module for finding biconnected components of the graph.

## How to install?

To build and install MAGE query modules you will need: **Python3**, **Make**, **CMake** and **Clang**.

### Installing with Docker

**1.** Make sure to have `memgraph:latest` Docker image.  
**2.** Build **MAGE** tagged Docker image.  
```
docker build . -t memgraph:mage
```

**3.** Start Memgraph with the following command and enjoy **MAGE**:
```
docker run -p 7687:7687 memgraph:mage
```

### Installing without Docker
**1.** Run the `build` script. It will generate a `dist` directory with all the needed files.  
```
python3 build
```

**2.** Copy the contents of the newly created `dist` directory to `/usr/lib/memgraph/query_modules`.  
**3.** Start Memgraph and enjoy **MAGE**!  

> Note that query modules are loaded into Memgraph on startup so if your instance was already running you will need to execute the following query to load them:
```
CALL mg.load_all();
```
If you want to find out more about loading query modules, visit [this guide](https://docs.memgraph.com/memgraph/database-functionalities/query-modules/load-call-query-modules).


## Testing the MAGE
To test that everything is built, loaded, and working correctly, a python script can be run. Make sure that the Memgraph instance with **MAGE** is up and running. 
```
# Running unit tests for C++ and Python
python3 test_unit

# Running end-to-end tests
python3 test_e2e
```
## Contributing

We encourage everyone to contribute with their own algorithm implementations and ideas. If you want to contribute or report a bug, please take a look at the [contributions guide](CONTRIBUTING.md).

## Code of Conduct

Everyone participating in this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to <tech@memgraph.com>.

## Feedback
Your feedback is always welcome and valuable to us. Please don't hesitate to post on our [Community Forum](https://discourse.memgraph.com/).
