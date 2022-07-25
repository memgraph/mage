#pragma once

#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "mg_procedure.h"
#include "mgp.hpp"

namespace mage {

class IndexException : public std::exception {
 public:
  explicit IndexException(const std::string &message) : message_(message){};
  const char *what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

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

namespace util {
// uint to int conversion in C++ is a bit tricky. Take a look here
// https://stackoverflow.com/questions/14623266/why-cant-i-reinterpret-cast-uint-to-int
// for more details.
template <typename TDest, typename TSrc>
TDest MemcpyCast(TSrc src) {
  TDest dest;
  static_assert(sizeof(dest) == sizeof(src), "MemcpyCast expects source and destination to be of same size");
  static_assert(std::is_arithmetic<TSrc>::value, "MemcpyCast expects source is an arithmetic type");
  static_assert(std::is_arithmetic<TDest>::value, "MemcpyCast expects destination is an arithmetic type");
  std::memcpy(&dest, &src, sizeof(src));
  return dest;
}
}  // namespace util

// Forward declarations
class ImmutableVertex;
class ImmutableEdge;
class ImmutableList;
class ImmutableValue;
class ImmutablePath;
class Vertex;
class Value;
class Vertices;
class Edges;
class Path;

#define CREATE_ITERATOR(container, element)                                                                         \
  class Iterator {                                                                                                  \
   private:                                                                                                         \
    friend class container;                                                                                         \
                                                                                                                    \
   public:                                                                                                          \
    bool operator==(const Iterator &other) const { return iterable_ == other.iterable_ && index_ == other.index_; } \
                                                                                                                    \
    bool operator!=(const Iterator &other) const { return !(*this == other); }                                      \
                                                                                                                    \
    Iterator &operator++() {                                                                                        \
      index_++;                                                                                                     \
      return *this;                                                                                                 \
    }                                                                                                               \
                                                                                                                    \
    element operator*() const;                                                                                      \
                                                                                                                    \
   private:                                                                                                         \
    Iterator(const container *iterable, size_t index) : iterable_(iterable), index_(index) {}                       \
                                                                                                                    \
    const container *iterable_;                                                                                     \
    size_t index_;                                                                                                  \
  }

/// Wrapper for int64_t ID to prevent dangerous implicit conversions.
class Id {
 public:
  Id() = default;

  /// Construct Id from uint64_t
  static Id FromUint(uint64_t id) { return Id(util::MemcpyCast<int64_t>(id)); }

  /// Construct Id from int64_t
  static Id FromInt(int64_t id) { return Id(id); }

  int64_t AsInt() const { return id_; }
  uint64_t AsUint() const { return util::MemcpyCast<uint64_t>(id_); }

  bool operator==(const Id &other) const { return id_ == other.id_; }
  bool operator!=(const Id &other) const { return !(*this == other); }

 private:
  explicit Id(int64_t id) : id_(id) {}

  int64_t id_;
};

////////////////////////////////////////////////////////////////////////////////
// Graph:

class Graph {
 public:
  explicit Graph(const mgp_graph *graph, mgp_memory *memory) : graph_(graph), memory_(memory){};

  Vertex GetVertexById(std::int64_t vertex_id);

  Vertices vertices() const;

  const mgp_graph *graph_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// List:

/// \brief Wrapper class for \ref mgp_list.
class List {
 private:
  friend class Value;
  friend class ImmutableList;

 public:
  CREATE_ITERATOR(List, ImmutableValue);

  explicit List(mgp_list *ptr, mgp_memory *memory) : ptr_(ptr), memory_(memory) {}

  List(List &&other);
  List &operator=(const List &other) = delete;
  List &operator=(List &&other) = delete;

  ~List();

  /// \brief Constructs a list that can hold at most \p capacity elements.
  /// \param capacity The maximum number of elements that the newly constructed
  ///                 list can hold.
  explicit List(size_t capacity, mgp_memory *memory) : List(mgp::list_make_empty(capacity, memory), memory) {}

  explicit List(const std::vector<Value> &values, mgp_memory *memory);

  explicit List(std::vector<Value> &&values, mgp_memory *memory);

  List(std::initializer_list<Value> list, mgp_memory *memory);

  size_t size() const { return mgp::list_size(ptr_); }

  bool empty() const { return size() == 0; }

  /// \brief Returns the value at the given `index`.
  const ImmutableValue operator[](size_t index) const;

  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

  /// \brief Appends the given `value` to the list.
  /// The `value` is copied.
  void Append(const Value &value);

  /// \brief Appends the given `value` to the list.
  /// The `value` is copied.
  void Append(const ImmutableValue &value);

  /// \brief Appends the given `value` to the list.
  /// \note
  /// It takes the ownership of the `value` by moving it.
  /// Behaviour of accessing the `value` after performing this operation is
  /// considered undefined.
  void Append(Value &&value);

  const ImmutableList AsImmutableList() const;

  /// \exception std::runtime_error list contains value with unknown type
  bool operator==(const List &other) const;
  /// \exception std::runtime_error list contains value with unknown type
  bool operator==(const ImmutableList &other) const;
  /// \exception std::runtime_error list contains value with unknown type
  bool operator!=(const List &other) const { return !(*this == other); }
  /// \exception std::runtime_error list contains value with unknown type
  bool operator!=(const ImmutableList &other) const { return !(*this == other); }

 private:
  mgp_list *ptr_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// ImmutableList:

class ImmutableList {
 public:
  friend class List;

  CREATE_ITERATOR(ImmutableList, ImmutableValue);

  explicit ImmutableList(const mgp_list *const_ptr, mgp_memory *memory) : const_ptr_(const_ptr), memory_(memory) {}

  size_t size() const { return mgp::list_size(const_cast<mgp_list *>(const_ptr_)); }
  bool empty() const { return size() == 0; }

  /// \brief Returns the value at the given `index`.
  const ImmutableValue operator[](size_t index) const;

  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

  /// \exception std::runtime_error list contains value with unknown type
  bool operator==(const ImmutableList &other) const;
  /// \exception std::runtime_error list contains value with unknown type
  bool operator==(const List &other) const;
  /// \exception std::runtime_error list contains value with unknown type
  bool operator!=(const ImmutableList &other) const { return !(*this == other); }
  /// \exception std::runtime_error list contains value with unknown type
  bool operator!=(const List &other) const { return !(*this == other); }

 private:
  const mgp_list *const_ptr_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// Properties:

class Properties {
 private:
  using KeyValuePair = std::pair<std::string_view, Value>;

 public:
  explicit Properties(mgp_properties_iterator *properties_iterator, mgp_memory *memory);

  size_t size() const { return property_map_.size(); }

  bool empty() const { return size() == 0; }

  /// \brief Returns the value associated with the given `key`.
  /// Behaves undefined if there is no such a value.
  /// \note
  /// Each key-value pair has to be checked, resulting with O(n)
  /// time complexity.
  ImmutableValue operator[](const std::string_view key) const;

  std::map<std::string_view, ImmutableValue>::const_iterator begin() const { return property_map_.begin(); }
  std::map<std::string_view, ImmutableValue>::const_iterator end() const { return property_map_.end(); }

  /// \brief Returns the key-value iterator for the given `key`.
  /// In the case there is no such pair, end iterator is returned.
  /// \note
  /// Each key-value pair has to be checked, resulting with O(n) time
  /// complexity.
  std::map<std::string_view, ImmutableValue>::const_iterator find(const std::string_view key) const {
    return property_map_.find(key);
  }

  /// \exception std::runtime_error map contains value with unknown type
  bool operator==(const Properties &other) const;
  /// \exception std::runtime_error map contains value with unknown type
  bool operator!=(const Properties &other) const { return !(*this == other); }

 private:
  std::map<const std::string_view, ImmutableValue> property_map_;
};

////////////////////////////////////////////////////////////////////////////////
// Labels:

/// \brief View of the node's labels
class Labels {
 public:
  CREATE_ITERATOR(Labels, std::string_view);

  explicit Labels(const mgp_vertex *vertex_ptr) : vertex_ptr_(vertex_ptr) {}

  size_t size() const { return mgp::vertex_labels_count(const_cast<mgp_vertex *>(vertex_ptr_)); }

  /// \brief Return node's label at the `index` position.
  std::string_view operator[](size_t index) const;

  Iterator begin() { return Iterator(this, 0); }
  Iterator end() { return Iterator(this, size()); }

 private:
  const mgp_vertex *vertex_ptr_;
};

////////////////////////////////////////////////////////////////////////////////
// Vertex:

/// \brief Wrapper class for \ref mgp_node
class Vertex {
 public:
  friend class ImmutableVertex;
  friend class Value;
  friend class Record;
  friend class Path;

  explicit Vertex(mgp_vertex *ptr, mgp_memory *memory) : ptr_(ptr), memory_(memory) {}

  /// \brief Create a Node from a copy of the given \ref mgp_node.
  explicit Vertex(const mgp_vertex *const_ptr, mgp_memory *memory)
      : Vertex(mgp::vertex_copy(const_cast<mgp_vertex *>(const_ptr), memory), memory) {}

  Vertex(const Vertex &other, mgp_memory *memory);
  Vertex(Vertex &&other);
  Vertex &operator=(const Vertex &other) = delete;
  Vertex &operator=(Vertex &&other) = delete;
  ~Vertex();

  explicit Vertex(const ImmutableVertex &vertex);

  Id id() const { return Id::FromInt(mgp::vertex_get_id(ptr_).as_int); }

  Labels labels() const { return Labels(ptr_); }

  Properties properties() const { return Properties(mgp::vertex_iter_properties(ptr_, memory_), memory_); }

  Edges in_edges() const;

  Edges out_edges() const;

  ImmutableVertex AsImmutableVertex() const;

  /// \exception std::runtime_error node property contains value with unknown type
  bool operator==(const Vertex &other) const;
  /// \exception std::runtime_error node property contains value with unknown type
  bool operator==(const ImmutableVertex &other) const;
  /// \exception std::runtime_error node property contains value with unknown type
  bool operator!=(const Vertex &other) const { return !(*this == other); }
  /// \exception std::runtime_error node property contains value with unknown type
  bool operator!=(const ImmutableVertex &other) const { return !(*this == other); }

 private:
  mgp_vertex *ptr_;
  mgp_memory *memory_;
};

class ImmutableVertex {
 public:
  friend class Vertex;
  friend class Value;
  friend class Record;
  friend class Path;

  explicit ImmutableVertex(const mgp_vertex *const_ptr, mgp_memory *memory) : const_ptr_(const_ptr), memory_(memory) {}

  Id id() const { return Id::FromInt(mgp::vertex_get_id(const_cast<mgp_vertex *>(const_ptr_)).as_int); }

  Labels labels() const { return Labels(const_ptr_); }

  Properties properties() const {
    return Properties(mgp::vertex_iter_properties(const_cast<mgp_vertex *>(const_ptr_), memory_), memory_);
  }

  Edges in_edges() const;

  Edges out_edges() const;

  /// \exception std::runtime_error node property contains value with unknown type
  bool operator==(const ImmutableVertex &other) const;
  /// \exception std::runtime_error node property contains value with unknown type
  bool operator==(const Vertex &other) const;
  /// \exception std::runtime_error node property contains value with unknown type
  bool operator!=(const ImmutableVertex &other) const { return !(*this == other); }
  /// \exception std::runtime_error node property contains value with
  bool operator!=(const Vertex &other) const { return !(*this == other); }

  const mgp_vertex *const_ptr_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// Value Type:

/// \brief Wrapper class for \ref mg_relationship.
class Edge {
 private:
  friend class Value;
  friend class ImmutableEdge;
  friend class Path;
  friend class Record;

 public:
  explicit Edge(mgp_edge *ptr, mgp_memory *memory) : ptr_(ptr), memory_(memory) {}

  /// \brief Create a Relationship from a copy the given \ref mg_relationship.
  explicit Edge(const mgp_edge *const_ptr, mgp_memory *memory)
      : Edge(mgp::edge_copy(const_cast<mgp_edge *>(const_ptr), memory), memory) {}

  Edge(const Edge &other);
  Edge(Edge &&other);
  Edge &operator=(const Edge &other) = delete;
  Edge &operator=(Edge &&other) = delete;
  ~Edge();

  explicit Edge(const ImmutableEdge &rel);

  Id id() const { return Id::FromInt(mgp::edge_get_id(ptr_).as_int); }

  /// \brief Return the Id of the node that is at the start of the relationship.
  ImmutableVertex from() const { return ImmutableVertex(mgp::edge_get_from(ptr_), memory_); }

  /// \brief Return the Id of the node that is at the end of the relationship.
  ImmutableVertex to() const { return ImmutableVertex(mgp::edge_get_to(ptr_), memory_); }

  std::string_view type() const;

  Properties properties() const { return Properties(mgp::edge_iter_properties(ptr_, memory_), memory_); }

  ImmutableEdge AsImmutableEdge() const;

  /// \exception std::runtime_error relationship property contains value with
  /// unknown type
  bool operator==(const Edge &other) const;
  /// \exception std::runtime_error relationship property contains value with
  /// unknown type
  bool operator==(const ImmutableEdge &other) const;
  /// \exception std::runtime_error relationship property contains value with
  /// unknown type
  bool operator!=(const Edge &other) const { return !(*this == other); }
  /// \exception std::runtime_error relationship property contains value with
  /// unknown type
  bool operator!=(const ImmutableEdge &other) const { return !(*this == other); }

 private:
  mgp_edge *ptr_;
  mgp_memory *memory_;
};

class ImmutableEdge {
 public:
  friend class Edge;
  friend class Record;
  friend class Path;

  explicit ImmutableEdge(const mgp_edge *const_ptr, mgp_memory *memory) : const_ptr_(const_ptr), memory_(memory) {}

  Id id() const { return Id::FromInt(mgp::edge_get_id(const_cast<mgp_edge *>(const_ptr_)).as_int); }

  /// \brief Return the Id of the node that is at the start of the relationship.
  ImmutableVertex from() const {
    return ImmutableVertex(mgp::edge_get_from(const_cast<mgp_edge *>(const_ptr_)), memory_);
  }

  /// \brief Return the Id of the node that is at the end of the relationship.
  ImmutableVertex to() const { return ImmutableVertex(mgp::edge_get_to(const_cast<mgp_edge *>(const_ptr_)), memory_); }

  std::string_view type() const;

  Properties properties() const {
    return Properties(mgp::edge_iter_properties(const_cast<mgp_edge *>(const_ptr_), memory_), memory_);
  }

  /// \exception std::runtime_error relationship property contains value with
  /// unknown type
  bool operator==(const ImmutableEdge &other) const;
  /// \exception std::runtime_error relationship property contains value with
  /// unknown type
  bool operator==(const Edge &other) const;
  /// \exception std::runtime_error relationship property contains value with
  /// unknown type
  bool operator!=(const ImmutableEdge &other) const { return !(*this == other); }
  /// \exception std::runtime_error relationship property contains value with
  /// unknown type
  bool operator!=(const Edge &other) const { return !(*this == other); }

 private:
  const mgp_edge *const_ptr_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// Path:
class Path final {
 private:
  friend class Value;
  friend class ImmutablePath;
  friend class Record;

 public:
  explicit Path(mgp_path *ptr, mgp_memory *memory) : ptr_(ptr), memory_(memory){};

  /// \brief Create a Path from a copy of the given \ref mg_path.
  explicit Path(const mgp_path *const_ptr, mgp_memory *memory)
      : Path(mgp::path_copy(const_cast<mgp_path *>(const_ptr), memory), memory_) {}

  explicit Path(const Vertex &start_vertex);

  explicit Path(const ImmutableVertex &start_vertex);

  Path(const Path &other);
  Path(Path &&other);
  Path &operator=(const Path &other);
  Path &operator=(Path &&other);
  ~Path();

  explicit Path(const ImmutablePath &path);

  /// Length of the path is number of edges.
  size_t length() const { return mgp::path_size(ptr_); }

  /// \brief Returns the vertex at the given `index`.
  /// \pre `index` should be less than or equal to length of the path.
  ImmutableVertex GetVertexAt(size_t index) const;

  /// \brief Returns the edge at the given `index`.
  /// \pre `index` should be less than length of the path.
  ImmutableEdge GetEdgeAt(size_t index) const;

  void Expand(const Edge &edge);

  void Expand(const ImmutableEdge &edge);

  ImmutablePath AsImmutablePath() const;

  /// \exception std::runtime_error path contains elements with unknown value
  bool operator==(const Path &other) const;
  /// \exception std::runtime_error path contains elements with unknown value
  bool operator==(const ImmutablePath &other) const;
  /// \exception std::runtime_error path contains elements with unknown value
  bool operator!=(const Path &other) const { return !(*this == other); }
  /// \exception std::runtime_error path contains elements with unknown value
  bool operator!=(const ImmutablePath &other) const { return !(*this == other); }

 private:
  mgp_path *ptr_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// Edges:
class ImmutablePath final {
 private:
  friend class Path;
  friend class Record;

 public:
  explicit ImmutablePath(const mgp_path *const_ptr, mgp_memory *memory) : const_ptr_(const_ptr), memory_(memory){};

  /// Length of the path in number of edges.
  size_t length() const { return mgp::path_size(const_cast<mgp_path *>(const_ptr_)); }

  /// \brief Returns the vertex at the given `index`.
  /// \pre `index` should be less than or equal to length of the path.
  ImmutableVertex GetVertexAt(size_t index) const;

  /// \brief Returns the edge at the given `index`.
  /// \pre `index` should be less than length of the path.
  ImmutableEdge GetEdgeAt(size_t index) const;

  /// \exception std::runtime_error path contains elements with unknown value
  bool operator==(const ImmutablePath &other) const;
  /// \exception std::runtime_error path contains elements with unknown value
  bool operator==(const Path &other) const;
  /// \exception std::runtime_error path contains elements with unknown value
  bool operator!=(const ImmutablePath &other) const { return !(*this == other); }
  /// \exception std::runtime_error path contains elements with unknown value
  bool operator!=(const Path &other) const { return !(*this == other); }

 private:
  const mgp_path *const_ptr_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// Edges:

class Edges {
 public:
  explicit Edges(mgp_edges_iterator *edges_iterator, mgp_memory *memory)
      : edges_iterator_(edges_iterator), memory_(memory){};

  class Iterator {
   public:
    friend class Edges;

    Iterator(mgp_edges_iterator *edges_iterator, mgp_memory *memory);
    ~Iterator();
    Iterator &operator++();
    Iterator operator++(int);
    bool operator==(Iterator other) const;
    bool operator!=(Iterator other) const { return !(*this == other); }
    ImmutableEdge operator*();
    // iterator traits
    using difference_type = ImmutableEdge;
    using value_type = ImmutableEdge;
    using pointer = const ImmutableEdge *;
    using reference = const ImmutableEdge &;
    using iterator_category = std::forward_iterator_tag;

   private:
    mgp_edges_iterator *edges_iterator_ = nullptr;
    mgp_memory *memory_;
    size_t index_ = 0;
  };

  Iterator begin();
  Iterator end();

 private:
  mgp_edges_iterator *edges_iterator_ = nullptr;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// Vertices:

class Vertices {
 public:
  explicit Vertices(mgp_vertices_iterator *vertices_iterator, mgp_memory *memory)
      : vertices_iterator_(vertices_iterator), memory_(memory){};

  class Iterator {
   public:
    friend class Vertices;

    Iterator(mgp_vertices_iterator *vertices_iterator, mgp_memory *memory);
    ~Iterator();
    Iterator &operator++();
    Iterator operator++(int);
    bool operator==(Iterator other) const;
    bool operator!=(Iterator other) const { return !(*this == other); }
    ImmutableVertex operator*();
    // iterator traits
    using difference_type = ImmutableVertex;
    using value_type = ImmutableVertex;
    using pointer = const ImmutableVertex *;
    using reference = const ImmutableVertex &;
    using iterator_category = std::forward_iterator_tag;

   private:
    mgp_memory *memory_;
    mgp_vertices_iterator *vertices_iterator_ = nullptr;
    size_t index_ = 0;
  };

  Iterator begin();
  Iterator end();

 private:
  mgp_vertices_iterator *vertices_iterator_ = nullptr;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// Value Type:

enum class ValueType : uint8_t {
  Null,
  Bool,
  Int,
  Double,
  String,
  List,
  Map,
  Vertex,
  Edge,
  Path,
};

////////////////////////////////////////////////////////////////////////////////
// Value:

/// Wrapper class for \ref mgp_value
class Value {
 public:
  friend class List;
  friend class Record;
  friend class ImmutableValue;

  explicit Value(mgp_value *ptr_, mgp_memory *memory) : ptr_(ptr_), memory_(memory){};
  ~Value();

  // Constructors for primitive types:
  explicit Value(bool value, mgp_memory *memory) : Value(mgp::value_make_bool(value, memory), memory){};
  explicit Value(int value, mgp_memory *memory) : Value(mgp::value_make_int(value, memory), memory){};
  explicit Value(int64_t value, mgp_memory *memory) : Value(mgp::value_make_int(value, memory), memory){};
  explicit Value(double value, mgp_memory *memory) : Value(mgp::value_make_double(value, memory), memory){};

  // Constructors for string:
  explicit Value(const std::string_view value, mgp_memory *memory)
      : Value(mgp::value_make_string(value.data(), memory), memory){};
  explicit Value(const char *value, mgp_memory *memory) : Value(mgp::value_make_string(value, memory), memory){};

  /// \brief Constructs a vertex value and takes the ownership of the given
  /// `vertex`. \note Behaviour of accessing the `vertex` after performing this
  /// operation is considered undefined.
  explicit Value(Vertex &&vertex, mgp_memory *memory) {
    Value(mgp::value_make_vertex(vertex.ptr_), memory);
    delete &vertex;
    vertex.ptr_ = nullptr;
  };

  /// \pre value type is Type::Bool
  bool ValueBool() const;
  /// \pre value type is Type::Int
  int64_t ValueInt() const;
  /// \pre value type is Type::Double
  double ValueDouble() const;
  /// \pre value type is Type::String
  std::string_view ValueString() const;
  /// \pre value type is Type::Node
  const ImmutableVertex ValueVertex() const;

  /// \exception std::runtime_error the value type is unknown
  ValueType type() const;

  /// \exception std::runtime_error the value type is unknown
  bool operator==(const Value &other) const;
  /// \exception std::runtime_error the value type is unknown
  bool operator==(const ImmutableValue &other) const;
  /// \exception std::runtime_error the value type is unknown
  bool operator!=(const Value &other) const { return !(*this == other); }
  /// \exception std::runtime_error the value type is unknown
  bool operator!=(const ImmutableValue &other) const { return !(*this == other); }

  const mgp_value *ptr() const { return ptr_; }

 private:
  mgp_value *ptr_;
  mgp_memory *memory_;
};

/// Wrapper class for \ref mgp_value
class ImmutableValue {
 public:
  friend class Properties;

  explicit ImmutableValue(const mgp_value *ptr_, mgp_memory *memory) : ptr_(ptr_), memory_(memory){};

  /// \pre value type is Type::Bool
  bool ValueBool() const;
  /// \pre value type is Type::Int
  int64_t ValueInt() const;
  /// \pre value type is Type::Double
  double ValueDouble() const;
  /// \pre value type is Type::String
  std::string_view ValueString() const;
  /// \pre value type is Type::Node
  const ImmutableVertex ValueVertex() const;

  /// \exception std::runtime_error the value type is unknown
  ValueType type() const;

  /// \exception std::runtime_error the value type is unknown
  bool operator==(const Value &other) const;
  /// \exception std::runtime_error the value type is unknown
  bool operator==(const ImmutableValue &other) const;
  /// \exception std::runtime_error the value type is unknown
  bool operator!=(const Value &other) const { return !(*this == other); }
  /// \exception std::runtime_error the value type is unknown
  bool operator!=(const ImmutableValue &other) const { return !(*this == other); }

 private:
  const mgp_value *ptr_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// Record:

class Record {
 public:
  explicit Record(mgp_result_record *record, mgp_memory *memory) : record_(record), memory_(memory){};

  void Insert(const char *field_name, const char *value);

  void Insert(const char *field_name, std::string_view value);

  void Insert(const char *field_name, std::int64_t value);

  void Insert(const char *field_name, double value);

  void Insert(const char *field_name, const Vertex &vertex);

  void Insert(const char *field_name, const ImmutableVertex &vertex);

  void Insert(const char *field_name, const Edge &edge);

  void Insert(const char *field_name, const ImmutableEdge &edge);

  void Insert(const char *field_name, const Path &path);

  void Insert(const char *field_name, const ImmutablePath &path);

 private:
  void Insert(const char *field_name, Value &&value);
  mgp_result_record *record_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// RecordFactory:

class RecordFactory {
 public:
  explicit RecordFactory(mgp_result *result, mgp_memory *memory) : result_(result), memory_(memory){};

  const mage::Record NewRecord() const;

  // TODO: Prevent implicit object creation
  // RecordFactory(RecordFactory const &) = delete;
  void operator=(RecordFactory const &) = delete;

 private:
  mgp_result *result_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////

namespace util {

inline bool ValuesEquals(const mgp_value *value1, const mgp_value *value2);

inline bool VerticesEquals(const mgp_vertex *node1, const mgp_vertex *node2) {
  // In query module scenario, vertices are same once they have similar ID
  if (node1 == node2) {
    return true;
  }
  if (mgp::vertex_get_id(const_cast<mgp_vertex *>(node1)).as_int !=
      mgp::vertex_get_id(const_cast<mgp_vertex *>(node2)).as_int) {
    return false;
  }
  return true;
}

inline bool EdgeEquals(const mgp_edge *edge1, const mgp_edge *edge2) {
  // In query module scenario, edges are same once they have similar ID
  if (edge1 == edge2) {
    return true;
  }
  if (mgp::edge_get_id(const_cast<mgp_edge *>(edge1)).as_int !=
      mgp::edge_get_id(const_cast<mgp_edge *>(edge2)).as_int) {
    return false;
  }
  return true;
}

inline bool PathEquals(const mgp_path *path1, const mgp_path *path2) {
  // In query module scenario, paths are same once they have similar ID
  if (path1 == path2) {
    return true;
  }
  if (mgp::path_size(const_cast<mgp_path *>(path1)) != mgp::path_size(const_cast<mgp_path *>(path2))) {
    return false;
  }
  const auto path_size = mgp::path_size(const_cast<mgp_path *>(path1));
  for (size_t i = 0; i < path_size; ++i) {
    if (!util::VerticesEquals(mgp::path_vertex_at(const_cast<mgp_path *>(path1), i),
                              mgp::path_vertex_at(const_cast<mgp_path *>(path2), i))) {
      return false;
    }
    if (!util::EdgeEquals(mgp::path_edge_at(const_cast<mgp_path *>(path1), i),
                          mgp::path_edge_at(const_cast<mgp_path *>(path2), i))) {
      return false;
    }
  }
  return util::VerticesEquals(mgp::path_vertex_at(const_cast<mgp_path *>(path1), path_size),
                              mgp::path_vertex_at(const_cast<mgp_path *>(path2), path_size));
}

inline bool ListEquals(const mgp_list *list1, const mgp_list *list2) {
  if (list1 == list2) {
    return true;
  }
  if (mgp::list_size(const_cast<mgp_list *>(list1)) != mgp::list_size(const_cast<mgp_list *>(list2))) {
    return false;
  }
  const size_t len = mgp::list_size(const_cast<mgp_list *>(list1));
  for (size_t i = 0; i < len; ++i) {
    if (!util::ValuesEquals(mgp::list_at(const_cast<mgp_list *>(list1), i),
                            mgp::list_at(const_cast<mgp_list *>(list2), i))) {
      return false;
    }
  }
  return true;
}

inline ValueType ConvertType(mgp_value_type type) {
  switch (type) {
    case MGP_VALUE_TYPE_NULL:
      return ValueType::Null;
    case MGP_VALUE_TYPE_BOOL:
      return ValueType::Bool;
    case MGP_VALUE_TYPE_INT:
      return ValueType::Int;
    case MGP_VALUE_TYPE_DOUBLE:
      return ValueType::Double;
    case MGP_VALUE_TYPE_STRING:
      return ValueType::String;
    case MGP_VALUE_TYPE_LIST:
      return ValueType::List;
    case MGP_VALUE_TYPE_MAP:
      return ValueType::Map;
    case MGP_VALUE_TYPE_VERTEX:
      return ValueType::Vertex;
    case MGP_VALUE_TYPE_EDGE:
      return ValueType::Edge;
    case MGP_VALUE_TYPE_PATH:
      return ValueType::Path;
    default:
      break;
  }
  throw ValueException("Unknown type error!");
}

inline bool ValuesEquals(const mgp_value *value1, const mgp_value *value2) {
  if (value1 == value2) {
    return true;
  }
  if (mgp::value_get_type(const_cast<mgp_value *>(value1)) != mgp::value_get_type(const_cast<mgp_value *>(value2))) {
    return false;
  }
  switch (mgp::value_get_type(const_cast<mgp_value *>(value1))) {
    case MGP_VALUE_TYPE_NULL:
      return true;
    case MGP_VALUE_TYPE_BOOL:
      return mgp::value_get_bool(const_cast<mgp_value *>(value1)) ==
             mgp::value_get_bool(const_cast<mgp_value *>(value2));
    case MGP_VALUE_TYPE_INT:
      return mgp::value_get_int(const_cast<mgp_value *>(value1)) == mgp::value_get_int(const_cast<mgp_value *>(value2));
    case MGP_VALUE_TYPE_DOUBLE:
      return mgp::value_get_double(const_cast<mgp_value *>(value1)) ==
             mgp::value_get_double(const_cast<mgp_value *>(value2));
    case MGP_VALUE_TYPE_STRING:
      return std::string_view(mgp::value_get_string(const_cast<mgp_value *>(value1))) ==
             std::string_view(mgp::value_get_string(const_cast<mgp_value *>(value2)));
    case MGP_VALUE_TYPE_LIST:
      return util::ListEquals(mgp::value_get_list(const_cast<mgp_value *>(value1)),
                              mgp::value_get_list(const_cast<mgp_value *>(value2)));
    // TODO: implement for maps
    case MGP_VALUE_TYPE_MAP:
      break;
    // return util::MapsEquals(mgp::value_map(value1), mgp::value_map(value2));
    case MGP_VALUE_TYPE_VERTEX:
      return util::VerticesEquals(mgp::value_get_vertex(const_cast<mgp_value *>(value1)),
                                  mgp::value_get_vertex(const_cast<mgp_value *>(value2)));
    case MGP_VALUE_TYPE_EDGE:
      return util::EdgeEquals(mgp::value_get_edge(const_cast<mgp_value *>(value1)),
                              mgp::value_get_edge(const_cast<mgp_value *>(value2)));
    // TODO: implement for Path
    case MGP_VALUE_TYPE_PATH:
      break;
    // TODO: the following
    case MGP_VALUE_TYPE_DATE:
      break;
    case MGP_VALUE_TYPE_LOCAL_TIME:
      break;
    case MGP_VALUE_TYPE_LOCAL_DATE_TIME:
      break;
    case MGP_VALUE_TYPE_DURATION:
      break;
  }
  throw ValueException("Value is invalid, it does not match any of Memgraph supported types.");
}
}  // namespace util

////////////////////////////////////////////////////////////////////////////////
// Graph:
inline Vertex Graph::GetVertexById(std::int64_t vertex_id) {
  auto vertex =
      mgp::graph_get_vertex_by_id(const_cast<mgp_graph *>(graph_), mgp_vertex_id{.as_int = vertex_id}, memory_);
  return Vertex(vertex, memory_);
}

inline Vertices Graph::vertices() const {
  auto *vertices_it = mgp::graph_iter_vertices(const_cast<mgp_graph *>(graph_), memory_);
  if (vertices_it == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }
  return Vertices(vertices_it, memory_);
}

////////////////////////////////////////////////////////////////////////////////
// Edges:

inline Edges::Iterator::Iterator(mgp_edges_iterator *edges_iterator, mgp_memory *memory)
    : edges_iterator_(edges_iterator), memory_(memory) {
  if (edges_iterator_ == nullptr) return;
  if (mgp::edges_iterator_get(edges_iterator_) == nullptr) {
    mgp::edges_iterator_destroy(edges_iterator_);
    edges_iterator_ = nullptr;
  }
}

inline Edges::Iterator::~Iterator() {
  if (edges_iterator_ != nullptr) {
    mgp::edges_iterator_destroy(edges_iterator_);
  }
}

inline Edges::Iterator &Edges::Iterator::operator++() {
  if (edges_iterator_ != nullptr) {
    auto next = mgp::edges_iterator_next(edges_iterator_);

    if (next == nullptr) {
      mgp::edges_iterator_destroy(edges_iterator_);
      edges_iterator_ = nullptr;
      return *this;
    }
    index_++;
  }
  return *this;
}

inline Edges::Iterator Edges::Iterator::operator++(int) {
  Edges::Iterator retval = *this;
  ++(*this);
  return retval;
}

inline bool Edges::Iterator::operator==(Iterator other) const {
  if (edges_iterator_ == nullptr && other.edges_iterator_ == nullptr) {
    return true;
  }
  if (edges_iterator_ == nullptr || other.edges_iterator_ == nullptr) {
    return false;
  }
  return mgp::edge_equal(mgp::edges_iterator_get(edges_iterator_), mgp::edges_iterator_get(other.edges_iterator_)) &&
         index_ == other.index_;
}

inline ImmutableEdge Edges::Iterator::operator*() {
  if (edges_iterator_ == nullptr) return ImmutableEdge(nullptr, memory_);

  auto vertex = ImmutableEdge(mgp::edges_iterator_get(edges_iterator_), memory_);
  return vertex;
}

inline Edges::Iterator Edges::begin() { return Iterator(edges_iterator_, memory_); }

inline Edges::Iterator Edges::end() { return Iterator(nullptr, memory_); }

////////////////////////////////////////////////////////////////////////////////
// Vertices:

inline Vertices::Iterator::Iterator(mgp_vertices_iterator *vertices_iterator, mgp_memory *memory)
    : memory_(memory), vertices_iterator_(vertices_iterator) {
  if (vertices_iterator_ == nullptr) return;
  if (mgp::vertices_iterator_get(vertices_iterator_) == nullptr) {
    mgp::vertices_iterator_destroy(vertices_iterator_);
    vertices_iterator_ = nullptr;
  }
}

inline Vertices::Iterator::~Iterator() {
  if (vertices_iterator_ != nullptr) {
    mgp::vertices_iterator_destroy(vertices_iterator_);
  }
}

inline Vertices::Iterator &Vertices::Iterator::operator++() {
  if (vertices_iterator_ != nullptr) {
    auto next = mgp::vertices_iterator_next(vertices_iterator_);

    if (next == nullptr) {
      mgp::vertices_iterator_destroy(vertices_iterator_);
      vertices_iterator_ = nullptr;
      return *this;
    }
    index_++;
  }
  return *this;
}

inline Vertices::Iterator Vertices::Iterator::operator++(int) {
  Vertices::Iterator retval = *this;
  ++(*this);
  return retval;
}
inline bool Vertices::Iterator::operator==(Iterator other) const {
  if (vertices_iterator_ == nullptr && other.vertices_iterator_ == nullptr) {
    return true;
  }
  if (vertices_iterator_ == nullptr || other.vertices_iterator_ == nullptr) {
    return false;
  }
  return mgp::vertex_equal(mgp::vertices_iterator_get(vertices_iterator_),
                           mgp::vertices_iterator_get(other.vertices_iterator_)) &&
         index_ == other.index_;
}

inline ImmutableVertex Vertices::Iterator::operator*() {
  if (vertices_iterator_ == nullptr) return ImmutableVertex(nullptr, memory_);

  auto vertex = ImmutableVertex(mgp::vertices_iterator_get(vertices_iterator_), memory_);
  return vertex;
}

inline Vertices::Iterator Vertices::begin() { return Iterator(vertices_iterator_, memory_); }

inline Vertices::Iterator Vertices::end() { return Iterator(nullptr, memory_); }

////////////////////////////////////////////////////////////////////////////////
// List:

inline ImmutableValue List::Iterator::operator*() const { return (*iterable_)[index_]; }

inline List::List(List &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; }

inline List::~List() {
  if (ptr_ != nullptr) {
    mgp::list_destroy(ptr_);
  }
}

inline List::List(const std::vector<Value> &values, mgp_memory *memory) : List(values.size(), memory) {
  for (const auto &value : values) {
    Append(value);
  }
}

inline List::List(std::vector<Value> &&values, mgp_memory *memory) : List(values.size(), memory) {
  for (auto &value : values) {
    Append(std::move(value));
  }
}

inline List::List(std::initializer_list<Value> values, mgp_memory *memory) : List(values.size(), memory) {
  for (const auto &value : values) {
    Append(value);
  }
}

inline const ImmutableValue List::operator[](size_t index) const {
  return ImmutableValue(mgp::list_at(ptr_, index), memory_);
}

// TODO: Implement safe value copying
// inline bool List::Append(const Value &value) { return mgp::list_append(ptr_, mgp::value_copy(value.ptr())) == 0; }

// inline bool List::Append(const ConstValue &value) { return mgp::list_append(ptr_, mgp::value_copy(value.ptr_)) == 0;
// }

inline void List::Append(Value &&value) {
  mgp::list_append(ptr_, value.ptr_);
  value.ptr_ = nullptr;
}

inline const ImmutableList List::AsImmutableList() const { return ImmutableList(ptr_, memory_); }

inline bool List::operator==(const List &other) const { return util::ListEquals(ptr_, other.ptr_); }

inline bool List::operator==(const ImmutableList &other) const { return util::ListEquals(ptr_, other.const_ptr_); }

inline ImmutableValue ImmutableList::Iterator::operator*() const { return (*iterable_)[index_]; }

inline const ImmutableValue ImmutableList::operator[](size_t index) const {
  return ImmutableValue(mgp::list_at(const_cast<mgp_list *>(const_ptr_), index), memory_);
}

inline bool ImmutableList::operator==(const ImmutableList &other) const {
  return util::ListEquals(const_ptr_, other.const_ptr_);
}

inline bool ImmutableList::operator==(const List &other) const { return util::ListEquals(const_ptr_, other.ptr_); }

////////////////////////////////////////////////////////////////////////////////
// Vertex:

inline std::string_view Labels::Iterator::operator*() const { return (*iterable_)[index_]; }

inline std::string_view Labels::operator[](size_t index) const {
  return mgp::vertex_label_at(const_cast<mgp_vertex *>(vertex_ptr_), index).name;
}

inline Vertex::Vertex(const Vertex &other, mgp_memory *memory) : Vertex(mgp::vertex_copy(other.ptr_, memory), memory) {}

inline Vertex::Vertex(Vertex &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; }

inline Vertex::~Vertex() {
  if (ptr_ != nullptr) {
    mgp::vertex_destroy(ptr_);
  }
}

inline Edges Vertex::in_edges() const {
  auto edge_iterator = mgp::vertex_iter_in_edges(ptr_, memory_);
  if (edge_iterator == nullptr) {
    throw NotEnoughMemoryException();
  }
  return Edges(edge_iterator, memory_);
}

inline Edges Vertex::out_edges() const {
  auto edge_iterator = mgp::vertex_iter_out_edges(ptr_, memory_);
  if (edge_iterator == nullptr) {
    throw NotEnoughMemoryException();
  }
  return Edges(edge_iterator, memory_);
}

inline bool Vertex::operator==(const Vertex &other) const { return util::VerticesEquals(ptr_, other.ptr_); }

inline bool Vertex::operator==(const ImmutableVertex &other) const {
  return util::VerticesEquals(ptr_, other.const_ptr_);
}

inline ImmutableVertex Vertex::AsImmutableVertex() const { return ImmutableVertex(ptr_, memory_); }

////////////////////////////////////////////////////////////////////////////////
// ImmutableVertex:

inline Edges ImmutableVertex::in_edges() const {
  auto edge_iterator = mgp::vertex_iter_in_edges(const_cast<mgp_vertex *>(const_ptr_), memory_);
  if (edge_iterator == nullptr) {
    throw NotEnoughMemoryException();
  }
  return Edges(edge_iterator, memory_);
}

inline Edges ImmutableVertex::out_edges() const {
  auto edge_iterator = mgp::vertex_iter_out_edges(const_cast<mgp_vertex *>(const_ptr_), memory_);
  if (edge_iterator == nullptr) {
    throw NotEnoughMemoryException();
  }
  return Edges(edge_iterator, memory_);
}

inline bool ImmutableVertex::operator==(const ImmutableVertex &other) const {
  return util::VerticesEquals(const_ptr_, other.const_ptr_);
}

inline bool ImmutableVertex::operator==(const Vertex &other) const {
  return util::VerticesEquals(const_ptr_, other.ptr_);
}

////////////////////////////////////////////////////////////////////////////////
// Properties:

inline Properties::Properties(mgp_properties_iterator *properties_iterator, mgp_memory *memory) {
  for (const auto *property = mgp::properties_iterator_get(properties_iterator); property;
       property = mgp::properties_iterator_next(properties_iterator)) {
    auto value = ImmutableValue(property->value, memory);
    property_map_.emplace(property->name, value);
  }
  mgp::properties_iterator_destroy(properties_iterator);
}

inline ImmutableValue Properties::operator[](const std::string_view key) const { return property_map_.at(key); }

inline bool Properties::operator==(const Properties &other) const { return property_map_ == other.property_map_; }

////////////////////////////////////////////////////////////////////////////////
// Value:

inline ValueType Value::type() const { return util::ConvertType(mgp::value_get_type(ptr_)); }

inline std::string_view Value::ValueString() const {
  if (type() != ValueType::String) {
    throw ValueException("Type of value is wrong: expected String.");
  }
  return mgp::value_get_string(ptr_);
}
inline bool Value::ValueBool() const {
  if (type() != ValueType::Bool) {
    throw ValueException("Type of value is wrong: expected Bool.");
  }
  return mgp::value_get_bool(ptr_);
}
inline std::int64_t Value::ValueInt() const {
  if (type() != ValueType::Int) {
    throw ValueException("Type of value is wrong: expected Int.");
  }
  return mgp::value_get_int(ptr_);
}
inline const ImmutableVertex Value::ValueVertex() const {
  if (type() != ValueType::Vertex) {
    throw ValueException("Type of value is wrong: expected Vertex.");
  }
  return ImmutableVertex(mgp::value_get_vertex(ptr_), memory_);
}

inline Value::~Value() {
  if (ptr_ != nullptr) {
    mgp::value_destroy(ptr_);
  }
}

////////////////////////////////////////////////////////////////////////////////
// ImmutableValue:

inline ValueType ImmutableValue::type() const {
  return util::ConvertType(mgp::value_get_type(const_cast<mgp_value *>(ptr_)));
}

inline std::string_view ImmutableValue::ValueString() const {
  if (type() != ValueType::String) {
    throw ValueException("Type of value is wrong: expected String.");
  }
  return mgp::value_get_string(const_cast<mgp_value *>(ptr_));
}
inline bool ImmutableValue::ValueBool() const {
  if (type() != ValueType::Bool) {
    throw ValueException("Type of value is wrong: expected Bool.");
  }
  return mgp::value_get_bool(const_cast<mgp_value *>(ptr_));
}
inline std::int64_t ImmutableValue::ValueInt() const {
  if (type() != ValueType::Int) {
    throw ValueException("Type of value is wrong: expected Int.");
  }
  return mgp::value_get_int(const_cast<mgp_value *>(ptr_));
}
inline const ImmutableVertex ImmutableValue::ValueVertex() const {
  if (type() != ValueType::Vertex) {
    throw ValueException("Type of value is wrong: expected Vertex.");
  }
  return ImmutableVertex(mgp::value_get_vertex(const_cast<mgp_value *>(ptr_)), memory_);
}

inline bool ImmutableValue::operator==(const ImmutableValue &other) const {
  return util::ValuesEquals(ptr_, other.ptr_);
}

inline bool ImmutableValue::operator==(const Value &other) const { return util::ValuesEquals(ptr_, other.ptr_); }

////////////////////////////////////////////////////////////////////////////////
// Edge:

inline Edge::Edge(const Edge &other) : Edge(mgp::edge_copy(other.ptr_, memory_), memory_) {}

inline Edge::Edge(Edge &&other) : Edge(other.ptr_, memory_) { other.ptr_ = nullptr; }

inline Edge::~Edge() {
  if (ptr_ != nullptr) {
    mgp::edge_destroy(ptr_);
  }
}

inline Edge::Edge(const ImmutableEdge &rel)
    : ptr_(mgp::edge_copy(const_cast<mgp_edge *>(rel.const_ptr_), memory_)), memory_(rel.memory_) {}

inline std::string_view Edge::type() const { return mgp::edge_get_type(ptr_).name; }

inline ImmutableEdge Edge::AsImmutableEdge() const { return ImmutableEdge(ptr_, memory_); }

inline bool Edge::operator==(const Edge &other) const { return util::EdgeEquals(ptr_, other.ptr_); }

inline bool Edge::operator==(const ImmutableEdge &other) const { return util::EdgeEquals(ptr_, other.const_ptr_); }

inline std::string_view ImmutableEdge::type() const {
  return mgp::edge_get_type(const_cast<mgp_edge *>(const_ptr_)).name;
}

inline bool ImmutableEdge::operator==(const ImmutableEdge &other) const {
  return util::EdgeEquals(const_ptr_, other.const_ptr_);
}

inline bool ImmutableEdge::operator==(const Edge &other) const { return util::EdgeEquals(const_ptr_, other.ptr_); }

////////////////////////////////////////////////////////////////////////////////
// Path:

inline Path::Path(const Path &other) : ptr_(mgp::path_copy(other.ptr_, other.memory_)), memory_(other.memory_){};

inline Path::Path(Path &&other) : ptr_(other.ptr_), memory_(other.memory_) { other.ptr_ = nullptr; }

inline Path::Path(const Vertex &start_vertex)
    : ptr_(mgp::path_make_with_start(start_vertex.ptr_, start_vertex.memory_)), memory_(start_vertex.memory_) {}

inline Path::Path(const ImmutableVertex &start_vertex)
    : ptr_(mgp::path_make_with_start(const_cast<mgp_vertex *>(start_vertex.const_ptr_), start_vertex.memory_)),
      memory_(start_vertex.memory_) {}

inline Path::~Path() {
  if (ptr_ != nullptr) {
    mgp::path_destroy(ptr_);
  }
}

inline Path::Path(const ImmutablePath &path)
    : ptr_(mgp::path_copy(const_cast<mgp_path *>(path.const_ptr_), path.memory_)), memory_(path.memory_) {}

inline ImmutableVertex Path::GetVertexAt(size_t index) const {
  auto vertex_ptr = mgp::path_vertex_at(ptr_, index);
  if (vertex_ptr == nullptr) {
    throw IndexException("Index value out of bounds.");
  }
  return ImmutableVertex(vertex_ptr, memory_);
}

inline ImmutableEdge Path::GetEdgeAt(size_t index) const {
  auto edge_ptr = mgp::path_edge_at(ptr_, index);
  if (edge_ptr == nullptr) {
    throw IndexException("Index value out of bounds.");
  }
  return ImmutableEdge(edge_ptr, memory_);
}

inline void Path::Expand(const Edge &edge) { mgp::path_expand(ptr_, edge.ptr_); }

inline void Path::Expand(const ImmutableEdge &edge) { mgp::path_expand(ptr_, const_cast<mgp_edge *>(edge.const_ptr_)); }

inline ImmutablePath Path::AsImmutablePath() const { return ImmutablePath(ptr_, memory_); }

inline bool Path::operator==(const Path &other) const { return util::PathEquals(ptr_, other.ptr_); }

inline bool Path::operator==(const ImmutablePath &other) const { return util::PathEquals(ptr_, other.const_ptr_); }

inline ImmutableVertex ImmutablePath::GetVertexAt(size_t index) const {
  auto vertex_ptr = mgp::path_vertex_at(const_cast<mgp_path *>(const_ptr_), index);
  if (vertex_ptr == nullptr) {
    throw IndexException("Index value out of bounds.");
  }
  return ImmutableVertex(vertex_ptr, memory_);
}

inline ImmutableEdge ImmutablePath::GetEdgeAt(size_t index) const {
  auto edge_ptr = mgp::path_edge_at(const_cast<mgp_path *>(const_ptr_), index);
  if (edge_ptr == nullptr) {
    throw IndexException("Index value out of bounds.");
  }
  return ImmutableEdge(edge_ptr, memory_);
}

inline bool ImmutablePath::operator==(const ImmutablePath &other) const {
  return util::PathEquals(const_ptr_, other.const_ptr_);
}

inline bool ImmutablePath::operator==(const Path &other) const { return util::PathEquals(const_ptr_, other.ptr_); }

////////////////////////////////////////////////////////////////////////////////
// Record:

inline void Record::Insert(const char *field_name, const char *value) { Insert(field_name, Value(value, memory_)); }

inline void Record::Insert(const char *field_name, std::string_view value) {
  Insert(field_name, Value(value, memory_));
}

inline void Record::Insert(const char *field_name, std::int64_t value) { Insert(field_name, Value(value, memory_)); }

inline void Record::Insert(const char *field_name, double value) { Insert(field_name, Value(value, memory_)); }

inline void Record::Insert(const char *field_name, const Vertex &vertex) {
  Insert(field_name, Value(mgp::value_make_vertex(mgp::vertex_copy(vertex.ptr_, vertex.memory_)), memory_));
}

inline void Record::Insert(const char *field_name, const ImmutableVertex &vertex) {
  Insert(field_name,
         Value(mgp::value_make_vertex(mgp::vertex_copy(const_cast<mgp_vertex *>(vertex.const_ptr_), vertex.memory_)),
               memory_));
}

inline void Record::Insert(const char *field_name, const Edge &edge) {
  Insert(field_name, Value(mgp::value_make_edge(mgp::edge_copy(edge.ptr_, edge.memory_)), memory_));
}

inline void Record::Insert(const char *field_name, const ImmutableEdge &edge) {
  Insert(field_name,
         Value(mgp::value_make_edge(mgp::edge_copy(const_cast<mgp_edge *>(edge.const_ptr_), edge.memory_)), memory_));
}

inline void Record::Insert(const char *field_name, const Path &path) {
  Insert(field_name, Value(mgp::value_make_path(mgp::path_copy(path.ptr_, path.memory_)), memory_));
}

inline void Record::Insert(const char *field_name, const ImmutablePath &path) {
  Insert(field_name,
         Value(mgp::value_make_path(mgp::path_copy(const_cast<mgp_path *>(path.const_ptr_), path.memory_)), memory_));
}

inline void Record::Insert(const char *field_name, Value &&value) {
  mgp::result_record_insert(record_, field_name, value.ptr_);
}

////////////////////////////////////////////////////////////////////////////////
// RecordFactory:

inline const Record RecordFactory::NewRecord() const {
  mgp_result_record *record = mgp::result_new_record(result_);
  if (record == nullptr) {
    throw NotEnoughMemoryException();
  }
  return Record(record, memory_);
}
}  // namespace mage