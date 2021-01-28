#include "mg_procedure.h"
#include "mg_utility.hpp"

/**
 * Convert Memgraph list to the edge patterns. Shared nodes are converted into edges.
 *
 * @param edge_patterns Label of shared nodes
 * @return List of patterns
 */
std::vector<std::vector<std::string>> ListToPattern(const mgp_list *edge_patterns) {
    std::vector<std::vector<std::string>> patterns;
    int size = mgp_list_size(edge_patterns);
    for (int i = 0; i < size; i++) {
        std::vector<std::string> pattern;

        const auto *pattern_list = mgp_value_get_list(mgp_list_at(edge_patterns, i));
        int pattern_size = mgp_list_size(pattern_list);
        for (int j = 0; j < pattern_size; j++) {
            std::string label = mgp_value_get_string(mgp_list_at(pattern_list, j));
            pattern.emplace_back(label);
        }
        patterns.emplace_back(pattern);
    }
    return patterns;
}

std::set<int64_t> GetAdjacentVertexId(const mgp_vertex *start_vertex, mgp_memory *memory) {
    std::set<int64_t> next_vertices;
    // Iterate on both sides
    auto *edges_it_out = mgp_vertex_iter_out_edges(start_vertex, memory);
    auto *edges_it_in = mgp_vertex_iter_in_edges(start_vertex, memory);
    if (edges_it_out == nullptr || edges_it_in == nullptr) {
        throw std::runtime_error("Not enough memory!");
    }

    // For each from, iterate through its edges
    for (const auto *out_edge = mgp_edges_iterator_get(edges_it_out); out_edge;
         out_edge = mgp_edges_iterator_next(edges_it_out)) {

        const mgp_vertex *to = mgp_edge_get_to(out_edge);
        if (start_vertex == to) continue;

        next_vertices.emplace(mgp_vertex_get_id(to).as_int);
    }
    // For each from, iterate through its edges
    for (const auto *in_edge = mgp_edges_iterator_get(edges_it_in); in_edge;
         in_edge = mgp_edges_iterator_next(edges_it_in)) {

        const mgp_vertex *from = mgp_edge_get_from(in_edge);
        if (start_vertex == from) continue;

        next_vertices.emplace(mgp_vertex_get_id(from).as_int);
    }
    // Destroy edge iterator
    mgp_edges_iterator_destroy(edges_it_out);
    mgp_edges_iterator_destroy(edges_it_in);

    return next_vertices;
}

std::set<int64_t> FindPattern(std::vector<std::string> pattern,
                              const mgp_vertex *start_vertex,
                              const mgp_graph *graph,
                              mgp_memory *memory) {

    int start_index = mgp_vertex_get_id(start_vertex).as_int;
    std::set<uint64_t> from_vertices;
    from_vertices.emplace(start_index);

    for (std::string label_name : pattern) {
        std::set<int64_t> next_vertices;

        for (int64_t from_id : from_vertices) {
            const mgp_vertex *from = mgp_graph_get_vertex_by_id(graph, mgp_vertex_id{.as_int = from_id}, memory);
            std::set<int64_t> adjacentID = GetAdjacentVertexId(from, memory);

            for (int64_t id: adjacentID) {
                const mgp_vertex *vertex = mgp_graph_get_vertex_by_id(graph, mgp_vertex_id{.as_int = id}, memory);
                bool label_check = mgp_vertex_has_label_named(vertex, label_name.c_str());
                if (label_check) next_vertices.emplace(id);
            }
        }

        from_vertices.clear();
        for (int64_t v: next_vertices) {
            from_vertices.emplace(v);
        }
        if (from_vertices.empty()) break;
    }

    std::set<int64_t> next_vertices;
    for (int64_t from_id : from_vertices) {
        const mgp_vertex *from = mgp_graph_get_vertex_by_id(graph, mgp_vertex_id{.as_int = from_id}, memory);
        std::set<int64_t> adjacentID = GetAdjacentVertexId(from, memory);

        for (int64_t id: adjacentID) {
            if (id != start_index) next_vertices.emplace(id);
        }
    }

    return next_vertices;
}

GraphMapping *MapMemgraphGraph(std::map<uint32_t, uint32_t> idToIter,
                               const mgp_graph *memgraphGraph,
                               mgp_memory *memory) {
    // Get vertices iterator
    mgp_vertices_iterator *vertices_iterator =
            mgp_graph_iter_vertices(memgraphGraph, memory);

    if (vertices_iterator == nullptr) {
        throw std::runtime_error("Not enough memory");
    }

    auto *vertices_it = mgp_graph_iter_vertices(memgraphGraph, memory);
    if (vertices_it == nullptr) {
        throw std::runtime_error("Not enough memory");
    }
    long NV = 0, NE = 0;
    // Iterating through vertices
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (const auto *vertex = mgp_vertices_iterator_get(vertices_it); vertex;
         vertex = mgp_vertices_iterator_next(vertices_it)) {
        auto *edges_it = mgp_vertex_iter_out_edges(vertex, memory);

        if (edges_it == nullptr) {
            mgp_vertices_iterator_destroy(vertices_it);
            throw std::runtime_error("Not enough memory");
        }

        // For each vertex, iterate through its edges
        for (const auto *out_edge = mgp_edges_iterator_get(edges_it); out_edge;
             out_edge = mgp_edges_iterator_next(edges_it)) {
            const mgp_vertex *from = vertex;
            const mgp_vertex *to = mgp_edge_get_to(out_edge);

            int64_t first_node_id = mgp_vertex_get_id(from).as_int;
            int64_t second_node_id = mgp_vertex_get_id(to).as_int;

            uint32_t first_node_map = idToIter[first_node_id];
            uint32_t second_node_map = idToIter[second_node_id];

            // Make list of edges
            edges.emplace_back(std::make_pair(first_node_map, second_node_map));
            NE++;
        }
        // Destroy edge iterator
        mgp_edges_iterator_destroy(edges_it);
        NV++;
    }
    // Destroy vertices iterator
    mgp_vertices_iterator_destroy(vertices_iterator);

    auto *mapping = new GraphMapping;
    mapping->numberOfVertices = NV;
    mapping->numberOfEdges = NE;
    mapping->edges = edges;
    return mapping;
}


GraphMapping *MapMemgraphGraphWithPatterns(const mgp_list *nodes,
                                           std::vector<std::vector<std::string>> patterns,
                                           std::map<uint32_t, uint32_t> idToIter,
                                           const mgp_graph *memgraphGraph,
                                           mgp_memory *memory) {
    long NV = 0, NE = 0;
    // Iterating through vertices
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    int nodes_size = mgp_list_size(nodes);
    for (int i = 0; i < nodes_size; i++) {
        const auto startVertex = mgp_value_get_vertex(mgp_list_at(nodes, i));

        // For each vertex, iterate through its edges
        const mgp_vertex *from = startVertex;
        for (std::vector<std::string> pattern : patterns) {
            std::set<int64_t> to_vertices = FindPattern(pattern, from, memgraphGraph, memory);
            if (to_vertices.empty()) continue;

            // Filter vertices that are available
            auto it = to_vertices.begin();
            while (it != to_vertices.end()) {
                if (!idToIter.count(*it)) {
                    it = to_vertices.erase(it);
                } else ++it;
            }

            for (int64_t to_id: to_vertices) {
                int64_t first_node_id = mgp_vertex_get_id(from).as_int;
                int64_t second_node_id = to_id;

                uint32_t first_node_map = idToIter[first_node_id];
                uint32_t second_node_map = idToIter[second_node_id];

                // Make list of edges
                if (first_node_map > second_node_map) continue;
                auto edge = std::make_pair(first_node_map, second_node_map);
                if (std::count(edges.begin(), edges.end(), edge)) continue;
                edges.emplace_back(edge);
                NE++;
            }
        }
        NV++;
    }
    auto *mapping = new GraphMapping;
    mapping->numberOfVertices = NV;
    mapping->numberOfEdges = NE;
    mapping->edges = edges;
    return mapping;
}


PairMap<uint32_t, uint32_t> VertexIdMappingSubgraph(const mgp_list *nodes,
                                            const mgp_graph *graph,
                                            mgp_result *result,
                                            mgp_memory *memory) {
    // Get vertices iterator
    auto *vertices_it = GetVerticesIterator(graph, result, memory);

    if (vertices_it == nullptr) {
        mgp_result_set_error_msg(result, "Not enough memory!");
        return std::nullopt;
    }

    std::map<uint32_t, uint32_t> iterToId;
    std::map<uint32_t, uint32_t> idToIter;

    uint32_t vertexIter = 0;
    int size = mgp_list_size(nodes);
    for (int i = 0; i < size; i++) {
        const auto *vertex = mgp_value_get_vertex(mgp_list_at(nodes, i));
        uint32_t vertexId = mgp_vertex_get_id(vertex).as_int;

        iterToId.insert(std::make_pair(vertexIter, vertexId));
        idToIter.insert(std::make_pair(vertexId, vertexIter));
        vertexIter++;
    }
    mgp_vertices_iterator_destroy(vertices_it);

    return std::make_pair(iterToId, idToIter);
}


mgp_vertices_iterator *GetVerticesIterator(const mgp_graph *graph,
                                           mgp_result *result,
                                           mgp_memory *memory) {
    mgp_vertices_iterator *vertices_iterator =
            mgp_graph_iter_vertices(graph, memory);

    if (vertices_iterator == nullptr) {
        mgp_result_set_error_msg(result, "Not enough memory");
        return nullptr;
    }

    auto *vertices_it = mgp_graph_iter_vertices(graph, memory);
    return vertices_it;
}


PairMap<uint32_t, uint32_t> VertexIdMapping(const mgp_graph *graph,
                                            mgp_result *result,
                                            mgp_memory *memory) {
    // Get vertices iterator
    auto *vertices_it = GetVerticesIterator(graph, result, memory);

    if (vertices_it == nullptr) {
        mgp_result_set_error_msg(result, "Not enough memory!");
        return std::nullopt;
    }

    std::map<uint32_t, uint32_t> iterToId;
    std::map<uint32_t, uint32_t> idToIter;

    uint32_t vertexIter = 0;
    for (const auto *vertex = mgp_vertices_iterator_get(vertices_it); vertex;
         vertex = mgp_vertices_iterator_next(vertices_it)) {
        uint32_t vertexId = mgp_vertex_get_id(vertex).as_int;

        iterToId.insert(std::make_pair(vertexIter, vertexId));
        idToIter.insert(std::make_pair(vertexId, vertexIter));
        vertexIter++;
    }
    mgp_vertices_iterator_destroy(vertices_it);

    return std::make_pair(iterToId, idToIter);
}
