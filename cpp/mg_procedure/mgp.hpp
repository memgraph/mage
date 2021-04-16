#pragma once

#include <cstring>
#include <map>
#include <set>
#include <string>

#include "mg_procedure.h"

namespace mgp {

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
class ImmutableValue;

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
// Properties:

class Properties final {
 private:
  using KeyValuePair = std::pair<std::string_view, ImmutableValue>;

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
  /// unknown type
  bool operator!=(const Vertex &other) const { return !(*this == other); }

 private:
  const mgp_vertex *const_ptr_;
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
class ImmutableValue final {
 public:
  explicit ImmutableValue(const mgp_value *const_ptr) : const_ptr_(const_ptr) {}

  /// \pre value type is Type::Bool
  bool ValueBool() const;
  /// \pre value type is Type::Int
  int64_t ValueInt() const;
  /// \pre value type is Type::Double
  double ValueDouble() const;
  /// \pre value type is Type::String
  std::string_view ValueString() const;

  /// \exception std::runtime_error the value type is unknown
  ValueType type() const;

  /// \exception std::runtime_error the value type is unknown
  bool operator==(const ImmutableValue &other) const;
  /// \exception std::runtime_error the value type is unknown
  bool operator!=(const ImmutableValue &other) const { return !(*this == other); }

  const mgp_value *ptr() const { return const_ptr_; }

 private:
  const mgp_value *const_ptr_;
};

namespace util {
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

}  // namespace util

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
    auto value = ImmutableValue(property->value);
    property_map_.emplace(property->name, value);
  }
  mgp_properties_iterator_destroy(properties_iterator);
}

inline ImmutableValue Properties::operator[](const std::string_view key) const { return property_map_.at(key); }

inline bool Properties::operator==(const Properties &other) const { return property_map_ == other.property_map_; }

////////////////////////////////////////////////////////////////////////////////
// ImmutableValue:

inline std::string_view ImmutableValue::ValueString() const { return mgp_value_get_string(const_ptr_); }
}  // namespace mgp
