# MAGE - Memgraph Advanced Graph Extensions

This open-source repository contains all available query modules written by the team behind Memgraph and its users. You can find and contribute implementations of various algorithms in multiple programming languages, all runnable inside Memgraph. This project aims to give everyone the tools they need to tackle the most challenging graph problems. 

## MAGE Query modules

Memgraph introduces the concept of query modules, user-defined procedures that extend the Cypher query language. These procedures are grouped into modules that can be loaded into Memgraph. You can find more information on query modules in the official [documentation](https://docs.memgraph.com/memgraph/database-functionalities/query-modules).

Query modules implemented in Python:
* [wcc.py](python/wcc.py): The WCC (Weakly Connected Components) query module can run WCC analysis on a sub-graph of the whole graph.
* [graph_analyzer.py](python/graph_analyzer.py): This Graph Analyzer query module offers insights about the stored graph or a subgraph.
* [nxalg](python/nxalg.py): A module that provides NetworkX integration with Memgraph and implements many NetworkX algorithms.  

Query modules implemented in C/C++:
* [connectivity_module.cpp](cpp/connectivity_module/connectivity_module.cpp): A module that finds weakly connected components in a graph.

## How to install?

To build and install MAGE query modules you will need:
*Make
*CMake
*Clang

Installing **with Docker**:

1. Run the `build.sh` script. It will generate a `dist` directory with all the needed files.
2. Run the following command where `dist` represent the path to your newly created `dist` directory:
```
docker volume create --driver local --opt type=none  --opt device=dist --opt o=bind dist
```
3. Start Memgraph with the following command:
```
docker run -it --rm -v dist:/usr/lib/memgraph/query_modules -p 7687:7687 memgraph
```

Installing **without Docker**:
1. Run the `build.sh` script. It will generate a `dist` directory with all the needed files.
2. Copy the contents of the newly created `dist` directory to `/usr/lib/memgraph/query_modules`.
3. Start Memgraph and enjoy **MAGE**.

Note that query modules are laoded into Memgraph on startup so if your instance was already running you will need to execute the following query to load them:
```
CALL mg.load_all();
```
If you want to find out more about loading query modules, visit [this guide](https://docs.memgraph.com/memgraph/database-functionalities/query-modules/load-call-query-modules).

## Contributing

We encourage everyone to contribute with their own algorithm implementations and ideas. If you want to contribute or report a bud, please take a look at the [contributions guide](CONTRIBUTING.md).

## Code of Conduct

Everyone participating in this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to <tech@memgraph.com>.

## Feedback
Your feedback is always welcome and valuable to us. Please don't hesitate to post on our [Community Forum](https://discourse.memgraph.com/).
