#include <functional>
#include <iostream>
#include <string>

#include "mg_procedure.h"

// TODO:jmatak Remove redundancy with other modules
namespace mgp {

class ValueException : public std::exception {
 public:
  explicit ValueException(const std::string &message) : message_(message){};
  const char *what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

class NotEnoughMemoryException : public std::exception {
 public:
  const char *what() const throw() { return "Not enough memory!"; }
};

class OnScopeExit {
 public:
  explicit OnScopeExit(const std::function<void()> &function) : function_(function) {}
  ~OnScopeExit() { function_(); }

 private:
  std::function<void()> function_;
};
}  // namespace mgp

namespace mgp {

class Vertex;
class Edge;

class Label {
 public:
  explicit Label(std::string name) : name_(name){};

  bool operator==(const Label &other) const { return name_ == other.name_; }

  std::string name_;
};

template <typename T>
class Properties {
 public:
  T vertex_or_edge_;
};

class Vertex {
  friend class Record;

 public:
  explicit Vertex() = delete;
  explicit Vertex(mgp_vertex *vertex) : vertex_(vertex){};
  explicit Vertex(const mgp_vertex *vertex) : vertex_(const_cast<mgp_vertex *>(vertex)){};

  std::uint64_t Id() const { return mgp_vertex_get_id(vertex_).as_int; }
  static Vertex *FromList(const mgp_list *list, int index) {
    return new Vertex(mgp_value_get_vertex(mgp_list_at(list, index)));
  }

  bool operator==(const mgp::Vertex &other) { return Id() == other.Id(); }

 private:
  mgp_vertex *vertex_;
};

class Edge {
  friend class Path;
  friend class Record;

 public:
  explicit Edge() = delete;
  explicit Edge(mgp_edge *edge) : edge_(edge) { SetVertices(); };
  explicit Edge(const mgp_edge *edge) : edge_(const_cast<mgp_edge *>(edge)) { SetVertices(); };

  std::uint64_t Id() const { return mgp_edge_get_id(edge_).as_int; }

  bool operator==(const mgp::Edge &other) { return Id() == other.Id(); }

  mgp::Vertex *FromVertex() const { return from_vertex_; }

  mgp::Vertex *ToVertex() const { return to_vertex_; }

 private:
  void SetVertices() {
    from_vertex_ = new mgp::Vertex(mgp_edge_get_from(edge_));
    to_vertex_ = new mgp::Vertex(mgp_edge_get_to(edge_));
  }

  mgp_edge *edge_;
  Vertex *from_vertex_;
  Vertex *to_vertex_;
};

class Vertices {
 public:
  explicit Vertices(const mgp_graph *graph, mgp_memory *memory) : graph_(graph), memory_(memory){};

  class Iterator {
   public:
    Iterator(mgp_vertices_iterator *vertices_iterator) {
      if (vertices_iterator != nullptr) {
        vertices_iterator_ = vertices_iterator;
        auto mgp_v = mgp_vertices_iterator_get(vertices_iterator);
        vertex_ = new mgp::Vertex(mgp_v);
      } else {
        vertex_ = nullptr;
      }
    };
    Iterator &operator++() {
      auto v = mgp_vertices_iterator_next(vertices_iterator_);
      if (v != nullptr) {
        vertex_ = new mgp::Vertex(v);
      } else {
        vertex_ = nullptr;
      }
      return *this;
    }
    Iterator operator++(int) {
      Iterator retval = *this;
      ++(*this);
      return retval;
    }
    bool operator==(Iterator other) const {
      if (other.vertex_ == nullptr && vertex_ == nullptr) return true;
      if (other.vertex_ == nullptr || vertex_ == nullptr) return false;
      return vertex_ == other.vertex_;
    }
    bool operator!=(Iterator other) const { return !(*this == other); }
    mgp::Vertex operator*() { return *vertex_; }
    // iterator traits
    using difference_type = mgp::Vertex;
    using value_type = mgp::Vertex;
    using pointer = const mgp::Vertex *;
    using reference = const mgp::Vertex &;
    using iterator_category = std::forward_iterator_tag;

   private:
    mgp::Vertex *vertex_;
    mgp_vertices_iterator *vertices_iterator_;
  };
  Iterator begin() {
    auto *vertices_it = mgp_graph_iter_vertices(graph_, memory_);
    if (vertices_it == nullptr) {
      throw mg_exception::NotEnoughMemoryException();
    }

    mgp::OnScopeExit delete_vertices_it([&vertices_it] {
      if (vertices_it != nullptr) {
        mgp_vertices_iterator_destroy(vertices_it);
      }
    });
    return Iterator(vertices_it);
  }
  Iterator end() { return Iterator(nullptr); }

 private:
  const mgp_graph *graph_;
  mgp_memory *memory_;
};

class Path {
  friend class Record;

 public:
  explicit Path(mgp_path *path) : path_(path) {
    auto path_size = mgp_path_size(path);
    // There is 1 edge fewer than number of vertices
    for (std::uint32_t i; i < path_size - 1; i++) {
      auto v = mgp_path_vertex_at(path, i);
      auto e = mgp_path_edge_at(path, i);

      vertices_.emplace_back(new mgp::Vertex(v));
      edges_.emplace_back(new mgp::Edge(e));
    }
    vertices_.emplace_back(new mgp::Vertex(mgp_path_vertex_at(path, path_size - 1)));
  };
  explicit Path(mgp_vertex *vertex, mgp_memory *memory) : path_(mgp_path_make_with_start(vertex, memory)) {
    vertices_.emplace_back(new mgp::Vertex(vertex));
  };

  void Expand(const mgp::Edge &edge) {
    auto last_vertex = vertices_[vertices_.size() - 1];
    if (last_vertex != edge.FromVertex()) {
      throw mgp::ValueException("Last vertex in edge is not part of given edge");
    }

    mgp_path_expand(path_, edge.edge_);
    vertices_.emplace_back(edge.ToVertex());
    edges_.emplace_back(&edge);
  }

  std::vector<const mgp::Vertex *> Vertices() { return vertices_; }

  std::vector<const mgp::Edge *> Edges() { return edges_; }

 private:
  mgp_path *path_;
  std::vector<const mgp::Vertex *> vertices_;
  std::vector<const mgp::Edge *> edges_;
};

///
///@brief Wrapper around state of the graph database
///
///
class Graph {
 public:
  explicit Graph(const mgp_graph *graph, mgp_memory *memory) : graph_(graph), memory_(memory){};

  mgp::Vertex GetVertexById(std::int64_t vertex_id) {
    auto vertex = mgp_graph_get_vertex_by_id(graph_, mgp_vertex_id{.as_int = vertex_id}, memory_);
    return Vertex(vertex);
  }

  mgp::Vertices Vertices() const { return mgp::Vertices(graph_, memory_); }

  const mgp_graph *graph_;
  mgp_memory *memory_;
};

class Record {
 public:
  explicit Record() = delete;
  explicit Record(mgp_result *result, mgp_memory *memory) : memory_(memory) {
    record_ = mgp_result_new_record(result);
    if (record_ == nullptr) {
      throw mgp::NotEnoughMemoryException();
    }
  }

  void Insert(const char *field_name, const mgp::Edge &edge_value) const {
    auto *value = mgp_value_make_edge(edge_value.edge_);
    if (value == nullptr) {
      throw mg_exception::NotEnoughMemoryException();
    }
    auto result_inserted = mgp_result_record_insert(record_, field_name, value);

    // TODO:jmatak Make memory management better
    // mgp_value_destroy(value);
    if (!result_inserted) {
      throw mg_exception::NotEnoughMemoryException();
    }
  }

  void Insert(const char *field_name, const mgp::Vertex &vertex_value) const {
    auto *value = mgp_value_make_vertex(vertex_value.vertex_);
    if (value == nullptr) {
      throw mg_exception::NotEnoughMemoryException();
    }
    auto result_inserted = mgp_result_record_insert(record_, field_name, value);

    // mgp_value_destroy(value);
    if (!result_inserted) {
      throw mg_exception::NotEnoughMemoryException();
    }
  }

  void Insert(const char *field_name, const char *string_value) const {
    auto *value = mgp_value_make_string(string_value, memory_);
    if (value == nullptr) {
      throw mg_exception::NotEnoughMemoryException();
    }
    auto result_inserted = mgp_result_record_insert(record_, field_name, value);

    // mgp_value_destroy(value);
    if (!result_inserted) {
      throw mg_exception::NotEnoughMemoryException();
    }
  }

  void Insert(const char *field_name, const int int_value) const {
    auto *value = mgp_value_make_int(int_value, memory_);
    if (value == nullptr) {
      throw mg_exception::NotEnoughMemoryException();
    }
    auto result_inserted = mgp_result_record_insert(record_, field_name, value);

    // mgp_value_destroy(value);
    if (!result_inserted) {
      throw mg_exception::NotEnoughMemoryException();
    }
  }

 private:
  mgp_result_record *record_;
  mgp_memory *memory_;
};

class RecordFactory {
 public:
  static RecordFactory &GetInstance(mgp_result *result, mgp_memory *memory) {
    static RecordFactory instance(result, memory);
    return instance;
  }

  mgp::Record *NewRecord() const {
    auto *record = new mgp::Record(result_, memory_);
    mgp::OnScopeExit delete_record([&record] { free(record); });
    return record;
  }

 public:
  // RecordFactory(RecordFactory const &) = delete;
  void operator=(RecordFactory const &) = delete;

 private:
  RecordFactory(mgp_result *result, mgp_memory *memory) : result_(result), memory_(memory){};
  mgp_result *result_;
  mgp_memory *memory_;
};

}  // namespace mgp