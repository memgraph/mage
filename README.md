# MAGE - Memgraph Advanced Graph Extensions

This open source repository contains all available query modules written by the team behind Memgraph and its users. You can find and contribute implementations of various algorithms in multiple programming languages, all runnable inside Memgraph. This project aims to give everyone the tools they need to tackle the most challanging graph problems. 

## Query modules

Memgraph introduces the concept of query modules, user-defined procedures that extend the Cypher query language. These procedures are grouped into modules which can be loaded into Memgraph. You can find more information on query modules in the official [documentation](https://docs.memgraph.com/memgraph/database-functionalities/query-modules).

## MAGE Contents

Query modules implemented in Python:
* [wcc.py](python/wcc.py): The WCC (Weakly Connected Components) query module can run WCC analysis on a sub-graph of the whole graph.
* [graph_analyzer.py](python/graph_analyzer.py): This Graph Analyzer query module offers insights about the stored graph or a subgraph.
* [nxalg](python/nxalg.py): A module that provides NetworkX integration with Memgraph and implements many NetworkX algorithms.  

Query modules implemented in C/C++:
* [connectivity_module.cpp](cpp/connectivity_module/connectivity_module.cpp): A module that finds weakly connected components in a graph.

## How to install?



## Contributing

We encourage everyone to contribute with their own algorithm imlpementations and ideas. If you want to contribute or report a bud, please take a look at the [contributions guide](CONTRIBUTING.md).

## Code of Conduct

This project and everyone participating in it is governed by this [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to <tech@memgraph.com>.