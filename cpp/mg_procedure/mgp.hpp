#pragma once

#include <cstring>
#include <map>
#include <set>
#include <string>

#include "mg_procedure.h"

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
class ImmutableValue;
class ImmutableList;
class Value;

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

/// \brief Wrapper class for \ref mgp_list.
class List final {
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
  explicit List(size_t capacity, mgp_memory *memory) : List(mgp_list_make_empty(capacity, memory), memory) {}

  explicit List(const std::vector<Value> &values, mgp_memory *memory);

  explicit List(std::vector<Value> &&values, mgp_memory *memory);

  List(std::initializer_list<Value> list, mgp_memory *memory);

  size_t size() const { return mgp_list_size(ptr_); }

  bool empty() const { return size() == 0; }

  /// \brief Returns the value at the given `index`.
  const ImmutableValue operator[](size_t index) const;

  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

  /// \brief Appends the given `value` to the list.
  /// The `value` is copied.
  bool Append(const Value &value);

  /// \brief Appends the given `value` to the list.
  /// The `value` is copied.
  bool Append(const ImmutableValue &value);

  /// \brief Appends the given `value` to the list.
  /// \note
  /// It takes the ownership of the `value` by moving it.
  /// Behaviour of accessing the `value` after performing this operation is
  /// considered undefined.
  bool Append(Value &&value);

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

class ImmutableList final {
 public:
  friend class List;

  CREATE_ITERATOR(ImmutableList, ImmutableValue);

  explicit ImmutableList(const mgp_list *const_ptr, mgp_memory *memory) : const_ptr_(const_ptr), memory_(memory) {}

  size_t size() const { return mgp_list_size(const_ptr_); }
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

class Properties final {
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
  Value operator[](const std::string_view key) const;

  std::map<std::string_view, Value>::const_iterator begin() const { return property_map_.begin(); }
  std::map<std::string_view, Value>::const_iterator end() const { return property_map_.end(); }

  /// \brief Returns the key-value iterator for the given `key`.
  /// In the case there is no such pair, end iterator is returned.
  /// \note
  /// Each key-value pair has to be checked, resulting with O(n) time
  /// complexity.
  std::map<std::string_view, Value>::const_iterator find(const std::string_view key) const {
    return property_map_.find(key);
  }

  /// \exception std::runtime_error map contains value with unknown type
  bool operator==(const Properties &other) const;
  /// \exception std::runtime_error map contains value with unknown type
  bool operator!=(const Properties &other) const { return !(*this == other); }

 private:
  std::map<const std::string_view, Value> property_map_;
};

////////////////////////////////////////////////////////////////////////////////
// Labels:

/// \brief View of the node's labels
class Labels final {
 public:
  CREATE_ITERATOR(Labels, std::string_view);

  explicit Labels(const mgp_vertex *vertex_ptr) : vertex_ptr_(vertex_ptr) {}

  size_t size() const { return mgp_vertex_labels_count(vertex_ptr_); }

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
class Vertex final {
 public:
  friend class ImmutableVertex;
  friend class Value;
  friend class Record;

  explicit Vertex(mgp_vertex *ptr, mgp_memory *memory) : ptr_(ptr), memory_(memory) {}

  /// \brief Create a Node from a copy of the given \ref mgp_node.
  explicit Vertex(const mgp_vertex *const_ptr, mgp_memory *memory)
      : Vertex(mgp_vertex_copy(const_ptr, memory), memory) {}

  Vertex(const Vertex &other, mgp_memory *memory);
  Vertex(Vertex &&other);
  Vertex &operator=(const Vertex &other) = delete;
  Vertex &operator=(Vertex &&other) = delete;
  ~Vertex();

  explicit Vertex(const ImmutableVertex &vertex);

  Id id() const { return Id::FromInt(mgp_vertex_get_id(ptr_).as_int); }

  Labels labels() const { return Labels(ptr_); }

  Properties properties() const { return Properties(mgp_vertex_iter_properties(ptr_, memory_), memory_); }

  ImmutableVertex AsConstVertex() const;

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

class ImmutableVertex final {
 public:
  friend class Vertex;
  friend class Value;

  explicit ImmutableVertex(const mgp_vertex *const_ptr, mgp_memory *memory) : const_ptr_(const_ptr), memory_(memory) {}

  Id id() const { return Id::FromInt(mgp_vertex_get_id(const_ptr_).as_int); }

  Labels labels() const { return Labels(const_ptr_); }

  Properties properties() const { return Properties(mgp_vertex_iter_properties(const_ptr_, memory_), memory_); }

  /// \exception std::runtime_error node property contains value with unknown type
  bool operator==(const ImmutableVertex &other) const;
  /// \exception std::runtime_error node property contains value with unknown type
  bool operator==(const Vertex &other) const;
  /// \exception std::runtime_error node property contains value with unknown type
  bool operator!=(const ImmutableVertex &other) const { return !(*this == other); }
  /// \exception std::runtime_error node property contains value with
  bool operator!=(const Vertex &other) const { return !(*this == other); }

 private:
  const mgp_vertex *const_ptr_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// Value Type:

/// \brief Wrapper class for \ref mg_relationship.
class Edge final {
 private:
  friend class Value;
  friend class ImmutableEdge;

 public:
  explicit Edge(mgp_edge *ptr, mgp_memory *memory) : ptr_(ptr), memory_(memory) {}

  /// \brief Create a Relationship from a copy the given \ref mg_relationship.
  explicit Edge(const mgp_edge *const_ptr, mgp_memory *memory) : Edge(mgp_edge_copy(const_ptr, memory), memory) {}

  Edge(const Edge &other);
  Edge(Edge &&other);
  Edge &operator=(const Edge &other) = delete;
  Edge &operator=(Edge &&other) = delete;
  ~Edge();

  explicit Edge(const ImmutableEdge &rel);

  Id id() const { return Id::FromInt(mgp_edge_get_id(ptr_).as_int); }

  /// \brief Return the Id of the node that is at the start of the relationship.
  Id from() const { return Id::FromInt(mgp_vertex_get_id(mgp_edge_get_from(ptr_)).as_int); }

  /// \brief Return the Id of the node that is at the end of the relationship.
  Id to() const { return Id::FromInt(mgp_vertex_get_id(mgp_edge_get_to(ptr_)).as_int); }

  std::string_view type() const;

  Properties properties() const { return Properties(mgp_edge_iter_properties(ptr_, memory_), memory_); }

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

class ImmutableEdge final {
 public:
  friend class Edge;

  explicit ImmutableEdge(const mgp_edge *const_ptr, mgp_memory *memory) : const_ptr_(const_ptr), memory_(memory) {}

  Id id() const { return Id::FromInt(mgp_edge_get_id(const_ptr_).as_int); }

  /// \brief Return the Id of the node that is at the start of the relationship.
  Id from() const { return Id::FromInt(mgp_vertex_get_id(mgp_edge_get_from(const_ptr_)).as_int); }

  /// \brief Return the Id of the node that is at the end of the relationship.
  Id to() const { return Id::FromInt(mgp_vertex_get_id(mgp_edge_get_to(const_ptr_)).as_int); }

  std::string_view type() const;

  Properties properties() const { return Properties(mgp_edge_iter_properties(const_ptr_, memory_), memory_); }

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
class Value final {
 public:
  friend class List;
  friend class Record;

  explicit Value(mgp_value *ptr_, mgp_memory *memory) : ptr_(ptr_), memory_(memory){};
  ~Value();

  // Constructors for primitive types:
  explicit Value(bool value, mgp_memory *memory) : Value(mgp_value_make_bool(value, memory), memory){};
  explicit Value(int value, mgp_memory *memory) : Value(mgp_value_make_int(value, memory), memory){};
  explicit Value(int64_t value, mgp_memory *memory) : Value(mgp_value_make_int(value, memory), memory){};
  explicit Value(double value, mgp_memory *memory) : Value(mgp_value_make_double(value, memory), memory){};

  // Constructors for string:
  explicit Value(const std::string_view value, mgp_memory *memory)
      : Value(mgp_value_make_string(value.data(), memory), memory){};
  explicit Value(const char *value, mgp_memory *memory) : Value(mgp_value_make_string(value, memory), memory){};

  /// \brief Constructs a vertex value and takes the ownership of the given
  /// `vertex`. \note Behaviour of accessing the `vertex` after performing this
  /// operation is considered undefined.
  explicit Value(Vertex &&vertex, mgp_memory *memory) {
    Value(mgp_value_make_vertex(vertex.ptr_), memory);
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
  bool operator!=(const Value &other) const { return !(*this == other); }

  const mgp_value *ptr() const { return ptr_; }

 private:
  mgp_value *ptr_;
  mgp_memory *memory_;
};

/// Wrapper class for \ref mgp_value
class ImmutableValue final {
 public:
  explicit ImmutableValue(const mgp_value *ptr_, mgp_memory *memory) : ptr_(ptr_), memory_(memory){};
  ~ImmutableValue();

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
  bool operator!=(const Value &other) const { return !(*this == other); }

  const mgp_value *ptr() const { return ptr_; }

 private:
  const mgp_value *ptr_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// Record:
class Record {
 public:
  explicit Record(mgp_result_record *record, mgp_memory *memory) : record_(record), memory_(memory){};

  ///
  ///@brief After inserting value into record, value is deleted
  ///
  ///@param field_name
  ///@param value
  ///
  void Insert(const char *field_name, const char *value);

  ///
  ///@brief After inserting value into record, value is deleted
  ///
  ///@param field_name
  ///@param value
  ///
  void Insert(const char *field_name, std::int64_t value);

  ///
  ///@brief After inserting value into record, value is deleted
  ///
  ///@param field_name
  ///@param value
  ///
  void Insert(const char *field_name, double value);

  ///
  ///@brief After inserting value into record, value is deleted
  ///
  ///@param field_name
  ///@param value
  ///
  void Insert(const char *field_name, const Vertex &value);

 private:
  void Insert(const char *field_name, const Value &value);
  mgp_result_record *record_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////
// RecordFactory:

class RecordFactory {
 public:
  static RecordFactory &GetInstance(mgp_result *result, mgp_memory *memory);

  const mgp::Record NewRecord() const;

  // TODO: Prevent implicit object creation
  // RecordFactory(RecordFactory const &) = delete;
  void operator=(RecordFactory const &) = delete;

 private:
  RecordFactory(mgp_result *result, mgp_memory *memory) : result_(result), memory_(memory){};
  mgp_result *result_;
  mgp_memory *memory_;
};

////////////////////////////////////////////////////////////////////////////////

namespace util {

inline bool AreValuesEqual(const mgp_value *value1, const mgp_value *value2);

inline bool VerticesEquals(const mgp_vertex *node1, const mgp_vertex *node2) {
  // In query module scenario, vertices are same once they have similar ID
  if (node1 == node2) {
    return true;
  }
  if (mgp_vertex_get_id(node1).as_int != mgp_vertex_get_id(node2).as_int) {
    return false;
  }
  return true;
}

inline bool EdgeEquals(const mgp_edge *edge1, const mgp_edge *edge2) {
  // In query module scenario, edges are same once they have similar ID
  if (edge1 == edge2) {
    return true;
  }
  if (mgp_edge_get_id(edge1).as_int != mgp_edge_get_id(edge2).as_int) {
    return false;
  }
  return true;
}

inline bool ListEquals(const mgp_list *list1, const mgp_list *list2) {
  if (list1 == list2) {
    return true;
  }
  if (mgp_list_size(list1) != mgp_list_size(list2)) {
    return false;
  }
  const size_t len = mgp_list_size(list1);
  for (size_t i = 0; i < len; ++i) {
    if (!util::AreValuesEqual(mgp_list_at(list1, i), mgp_list_at(list2, i))) {
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

inline bool AreValuesEqual(const mgp_value *value1, const mgp_value *value2) {
  if (value1 == value2) {
    return true;
  }
  if (mgp_value_get_type(value1) != mgp_value_get_type(value2)) {
    return false;
  }
  switch (mgp_value_get_type(value1)) {
    case MGP_VALUE_TYPE_NULL:
      return true;
    case MGP_VALUE_TYPE_BOOL:
      return mgp_value_get_bool(value1) == mgp_value_get_bool(value2);
    case MGP_VALUE_TYPE_INT:
      return mgp_value_get_int(value1) == mgp_value_get_int(value2);
    case MGP_VALUE_TYPE_DOUBLE:
      return mgp_value_get_double(value1) == mgp_value_get_double(value2);
    case MGP_VALUE_TYPE_STRING:
      return std::string_view(mgp_value_get_string(value1)) == std::string_view(mgp_value_get_string(value2));
    case MGP_VALUE_TYPE_LIST:
      return util::ListEquals(mgp_value_get_list(value1), mgp_value_get_list(value2));
    // TODO: implement for maps
    // case MGP_VALUE_TYPE_MAP:
    // return util::MapsEquals(mgp_value_map(value1), mgp_value_map(value2));
    case MGP_VALUE_TYPE_VERTEX:
      return util::VerticesEquals(mgp_value_get_vertex(value1), mgp_value_get_vertex(value2));
    case MGP_VALUE_TYPE_EDGE:
      return util::EdgeEquals(mgp_value_get_edge(value1), mgp_value_get_edge(value2));
  }
  throw ValueException("Value is invalid, it does not match any of Memgraph supported types.");
}
}  // namespace util

////////////////////////////////////////////////////////////////////////////////
// List:

inline ImmutableValue List::Iterator::operator*() const { return (*iterable_)[index_]; }

inline List::List(List &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; }

inline List::~List() {
  if (ptr_ != nullptr) {
    mgp_list_destroy(ptr_);
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
  return ImmutableValue(mgp_list_at(ptr_, index), memory_);
}

// TODO: Implement safe value copying
// inline bool List::Append(const Value &value) { return mgp_list_append(ptr_, mgp_value_copy(value.ptr())) == 0; }

// inline bool List::Append(const ConstValue &value) { return mgp_list_append(ptr_, mgp_value_copy(value.ptr_)) == 0; }

inline bool List::Append(Value &&value) {
  bool result = mgp_list_append(ptr_, value.ptr_) == 0;
  value.ptr_ = nullptr;
  return result;
}

inline const ImmutableList List::AsImmutableList() const { return ImmutableList(ptr_, memory_); }

inline bool List::operator==(const List &other) const { return util::ListEquals(ptr_, other.ptr_); }

inline bool List::operator==(const ImmutableList &other) const { return util::ListEquals(ptr_, other.const_ptr_); }

inline ImmutableValue ImmutableList::Iterator::operator*() const { return (*iterable_)[index_]; }

inline const ImmutableValue ImmutableList::operator[](size_t index) const {
  return ImmutableValue(mgp_list_at(const_ptr_, index), memory_);
}

inline bool ImmutableList::operator==(const ImmutableList &other) const {
  return util::ListEquals(const_ptr_, other.const_ptr_);
}

inline bool ImmutableList::operator==(const List &other) const { return util::ListEquals(const_ptr_, other.ptr_); }

////////////////////////////////////////////////////////////////////////////////
// Vertex:

inline std::string_view Labels::Iterator::operator*() const { return (*iterable_)[index_]; }

inline std::string_view Labels::operator[](size_t index) const { return mgp_vertex_label_at(vertex_ptr_, index).name; }

inline Vertex::Vertex(const Vertex &other, mgp_memory *memory) : Vertex(mgp_vertex_copy(other.ptr_, memory), memory) {}

inline Vertex::Vertex(Vertex &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; }

inline Vertex::~Vertex() {
  if (ptr_ != nullptr) {
    mgp_vertex_destroy(ptr_);
  }
}

inline bool Vertex::operator==(const Vertex &other) const { return util::VerticesEquals(ptr_, other.ptr_); }

inline bool Vertex::operator==(const ImmutableVertex &other) const {
  return util::VerticesEquals(ptr_, other.const_ptr_);
}

// inline ImmutableVertex Vertex::AsConstVertex() const { return ImmutableVertex(ptr_); }

inline bool ImmutableVertex::operator==(const ImmutableVertex &other) const {
  return util::VerticesEquals(const_ptr_, other.const_ptr_);
}

inline bool ImmutableVertex::operator==(const Vertex &other) const {
  return util::VerticesEquals(const_ptr_, other.ptr_);
}

////////////////////////////////////////////////////////////////////////////////
// Properties:

inline Properties::Properties(mgp_properties_iterator *properties_iterator, mgp_memory *memory) {
  for (const auto *property = mgp_properties_iterator_get(properties_iterator); property;
       property = mgp_properties_iterator_next(properties_iterator)) {
    auto value = Value(property->value, memory);
    property_map_.emplace(property->name, value);
  }
  mgp_properties_iterator_destroy(properties_iterator);
}

inline Value Properties::operator[](const std::string_view key) const { return property_map_.at(key); }

inline bool Properties::operator==(const Properties &other) const { return property_map_ == other.property_map_; }

////////////////////////////////////////////////////////////////////////////////
// Value:

inline ValueType Value::type() const { return util::ConvertType(mgp_value_get_type(ptr_)); }

inline std::string_view Value::ValueString() const {
  if (type() != ValueType::String) {
    throw ValueException("Type of value is wrong: expected String.");
  }
  return mgp_value_get_string(ptr_);
}
inline bool Value::ValueBool() const {
  if (type() != ValueType::Bool) {
    throw ValueException("Type of value is wrong: expected Bool.");
  }
  return mgp_value_get_bool(ptr_);
}
inline std::int64_t Value::ValueInt() const {
  if (type() != ValueType::Int) {
    throw ValueException("Type of value is wrong: expected Int.");
  }
  return mgp_value_get_int(ptr_);
}
inline const ImmutableVertex Value::ValueVertex() const {
  if (type() != ValueType::Vertex) {
    throw ValueException("Type of value is wrong: expected Vertex.");
  }
  return ImmutableVertex(mgp_value_get_vertex(ptr_), memory_);
}

inline Value::~Value() {
  if (ptr_ != nullptr) {
    mgp_value_destroy(ptr_);
  }
}

inline ValueType ImmutableValue::type() const { return util::ConvertType(mgp_value_get_type(ptr_)); }

inline std::string_view ImmutableValue::ValueString() const {
  if (type() != ValueType::String) {
    throw ValueException("Type of value is wrong: expected String.");
  }
  return mgp_value_get_string(ptr_);
}
inline bool ImmutableValue::ValueBool() const {
  if (type() != ValueType::Bool) {
    throw ValueException("Type of value is wrong: expected Bool.");
  }
  return mgp_value_get_bool(ptr_);
}
inline std::int64_t ImmutableValue::ValueInt() const {
  if (type() != ValueType::Int) {
    throw ValueException("Type of value is wrong: expected Int.");
  }
  return mgp_value_get_int(ptr_);
}
inline const ImmutableVertex ImmutableValue::ValueVertex() const {
  if (type() != ValueType::Vertex) {
    throw ValueException("Type of value is wrong: expected Vertex.");
  }
  return ImmutableVertex(mgp_value_get_vertex(ptr_), memory_);
}

////////////////////////////////////////////////////////////////////////////////
// Edge:

inline Edge::Edge(const Edge &other) : Edge(mgp_edge_copy(other.ptr_, memory_), memory_) {}

inline Edge::Edge(Edge &&other) : Edge(other.ptr_, memory_) { other.ptr_ = nullptr; }

inline Edge::~Edge() {
  if (ptr_ != nullptr) {
    mgp_edge_destroy(ptr_);
  }
}

inline Edge::Edge(const ImmutableEdge &rel) : ptr_(mgp_edge_copy(rel.const_ptr_, memory_)), memory_(rel.memory_) {}

inline std::string_view Edge::type() const { return mgp_edge_get_type(ptr_).name; }

inline ImmutableEdge Edge::AsImmutableEdge() const { return ImmutableEdge(ptr_, memory_); }

inline bool Edge::operator==(const Edge &other) const { return util::EdgeEquals(ptr_, other.ptr_); }

inline bool Edge::operator==(const ImmutableEdge &other) const { return util::EdgeEquals(ptr_, other.const_ptr_); }

inline std::string_view ImmutableEdge::type() const { return mgp_edge_get_type(const_ptr_).name; }

inline bool ImmutableEdge::operator==(const ImmutableEdge &other) const {
  return util::EdgeEquals(const_ptr_, other.const_ptr_);
}

inline bool ImmutableEdge::operator==(const Edge &other) const { return util::EdgeEquals(const_ptr_, other.ptr_); }

////////////////////////////////////////////////////////////////////////////////
// Record:

inline void Record::Insert(const char *field_name, const char *value) { Insert(field_name, Value(value, memory_)); }

inline void Record::Insert(const char *field_name, std::int64_t value) { Insert(field_name, Value(value, memory_)); }

inline void Record::Insert(const char *field_name, double value) { Insert(field_name, Value(value, memory_)); }

inline void Record::Insert(const char *field_name, const Vertex &vertex) {
  mgp_value *value = mgp_value_make_vertex(vertex.ptr_);
  if (value == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }
  auto result_inserted = mgp_result_record_insert(record_, field_name, value);
  if (!result_inserted) {
    throw mg_exception::NotEnoughMemoryException();
  }
}

inline void Record::Insert(const char *field_name, const Value &value) {
  auto result_inserted = mgp_result_record_insert(record_, field_name, value.ptr_);
  if (!result_inserted) {
    throw NotEnoughMemoryException();
  };
}

////////////////////////////////////////////////////////////////////////////////
// RecordFactory:

inline RecordFactory &RecordFactory::GetInstance(mgp_result *result, mgp_memory *memory) {
  static RecordFactory instance(result, memory);
  return instance;
}

inline const Record RecordFactory::NewRecord() const {
  mgp_result_record *record = mgp_result_new_record(result_);
  if (record == nullptr) {
    throw NotEnoughMemoryException();
  }
  return Record(record, memory_);
}
}  // namespace mgp
