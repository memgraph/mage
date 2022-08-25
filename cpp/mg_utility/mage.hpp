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
  static_assert(sizeof(dest) == sizeof(src), "MemcpyCast expects source and destination to be of the same size");
  static_assert(std::is_arithmetic<TSrc>::value, "MemcpyCast expects source to be an arithmetic type");
  static_assert(std::is_arithmetic<TDest>::value, "MemcpyCast expects destination to be an arithmetic type");
  std::memcpy(&dest, &src, sizeof(src));
  return dest;
}
}  // namespace util

// Forward declarations
class Vertices;
using GraphVertices = Vertices;
class GraphEdges;
class Edges;
class Vertex;
class Edge;
struct MapItem;
class Duration;
class Value;

mgp_memory *memory;

/* #region Graph (Id, Graph, Vertices, GraphEdges, Edges, Properties & Labels) */

/// Wrapper for int64_t IDs to prevent dangerous implicit conversions.
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

  bool operator<(const Id &other) const { return id_ < other.id_; }

 private:
  explicit Id(int64_t id) : id_(id) {}

  int64_t id_;
};

class Graph {
 public:
  explicit Graph(mgp_graph *graph) : graph_(graph) {}

  /// \brief Returns the graph order (number of vertices).
  int64_t order() const;
  /// \brief Returns the graph size (number of edges).
  int64_t size() const;

  GraphVertices vertices() const;
  GraphEdges edges() const;

  Vertex GetVertexById(const Id vertex_id) const;

  bool Contains(const Id vertex_id) const;
  bool Contains(const Vertex &vertex) const;
  bool Contains(const Edge &edge) const;

 private:
  mgp_graph *graph_;
};

class Vertices {
 public:
  explicit Vertices(mgp_vertices_iterator *vertices_iterator) : vertices_iterator_(vertices_iterator){};

  class Iterator {
   public:
    friend class Vertices;

    explicit Iterator(mgp_vertices_iterator *vertices_iterator);
    ~Iterator();
    Iterator &operator++();
    Iterator operator++(int);
    bool operator==(Iterator other) const;
    bool operator!=(Iterator other) const { return !(*this == other); }
    Vertex operator*();
    // iterator traits
    using difference_type = Vertex;
    using value_type = Vertex;
    using pointer = const Vertex *;
    using reference = const Vertex &;
    using iterator_category = std::forward_iterator_tag;

   private:
    mgp_vertices_iterator *vertices_iterator_ = nullptr;
    size_t index_ = 0;
  };

  Iterator begin();
  Iterator end();

 private:
  mgp_vertices_iterator *vertices_iterator_ = nullptr;
};

class GraphEdges {
 public:
  explicit GraphEdges(mgp_graph *graph) : graph_(graph){};

  class Iterator {
   public:
    friend class GraphEdges;

    explicit Iterator(mgp_vertices_iterator *vertices_iterator);
    ~Iterator();
    Iterator &operator++();
    bool operator==(Iterator other) const;
    bool operator!=(Iterator other) const { return !(*this == other); }
    Edge operator*();
    // iterator traits
    using difference_type = Edge;
    using value_type = Edge;
    using pointer = const Edge *;
    using reference = const Edge &;
    using iterator_category = std::forward_iterator_tag;

   private:
    mgp_vertices_iterator *vertices_iterator_ = nullptr;
    mgp_edges_iterator *out_edges_iterator_ = nullptr;
    size_t index_ = 0;
  };

  Iterator begin();
  Iterator end();

 private:
  mgp_graph *graph_;
};

class Edges {
 public:
  explicit Edges(mgp_edges_iterator *edges_iterator) : edges_iterator_(edges_iterator){};

  class Iterator {
   public:
    friend class Edges;

    explicit Iterator(mgp_edges_iterator *edges_iterator);
    ~Iterator();
    Iterator &operator++();
    Iterator operator++(int);
    bool operator==(Iterator other) const;
    bool operator!=(Iterator other) const { return !(*this == other); }
    Edge operator*();
    // iterator traits
    using difference_type = Edge;
    using value_type = Edge;
    using pointer = const Edge *;
    using reference = const Edge &;
    using iterator_category = std::forward_iterator_tag;

   private:
    mgp_edges_iterator *edges_iterator_ = nullptr;
    size_t index_ = 0;
  };

  Iterator begin();
  Iterator end();

 private:
  mgp_edges_iterator *edges_iterator_ = nullptr;
};

/// \brief View of node properties.
class Properties {
 public:
  explicit Properties(mgp_properties_iterator *properties_iterator);

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

/// \brief View of node labels.
class Labels {
 public:
  explicit Labels(mgp_vertex *vertex_ptr) : vertex_ptr_(vertex_ptr) {}

  size_t size() const { return mgp::vertex_labels_count(vertex_ptr_); }

  /// \brief Return node’s label at position `index`.
  std::string_view operator[](size_t index) const;

  class Iterator {
   private:
    friend class Labels;

   public:
    bool operator==(const Iterator &other) const { return iterable_ == other.iterable_ && index_ == other.index_; }

    bool operator!=(const Iterator &other) const { return !(*this == other); }

    Iterator &operator++() {
      index_++;
      return *this;
    }

    std::string_view operator*() const;

   private:
    Iterator(const Labels *iterable, size_t index) : iterable_(iterable), index_(index) {}

    const Labels *iterable_;
    size_t index_;
  };

  Iterator begin() { return Iterator(this, 0); }
  Iterator end() { return Iterator(this, size()); }

 private:
  mgp_vertex *vertex_ptr_;
};
/* #endregion */

/* #region Types */

/* #region Containers (List, Map) */

/// \brief Wrapper class for \ref mgp_list.
class List {
 private:
  friend class Value;
  friend class Record;

 public:
  explicit List(mgp_list *ptr) : ptr_(ptr) {}

  List &operator=(const List &other) = delete;
  List &operator=(List &&other) = delete;

  size_t size() const { return mgp::list_size(ptr_); }
  bool empty() const { return size() == 0; }

  /// \brief Returns the value at the given `index`.
  const Value operator[](size_t index) const;

  class Iterator {
   private:
    friend class List;

   public:
    bool operator==(const Iterator &other) const { return iterable_ == other.iterable_ && index_ == other.index_; }

    bool operator!=(const Iterator &other) const { return !(*this == other); }

    Iterator &operator++() {
      index_++;
      return *this;
    }

    Value operator*() const;

   private:
    Iterator(const List *iterable, size_t index) : iterable_(iterable), index_(index) {}

    const List *iterable_;
    size_t index_;
  };

  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, size()); }

  /// \brief Appends the given `value` to the list.
  /// The `value` is copied.
  void Append(const Value &value);
  /// \brief Appends the given `value` to the list.
  /// \note
  /// It takes the ownership of the `value` by moving it.
  /// Behaviour of accessing the `value` after performing this operation is
  /// considered undefined.
  void Append(Value &&value);

  void AppendExtend(Value &value);

  // Value Pop(); // (requires mgp_list_pop in the MGP API):

  /// \exception std::runtime_error list contains value with unknown type
  bool operator==(const List &other) const;
  /// \exception std::runtime_error list contains value with unknown type
  bool operator!=(const List &other) const { return !(*this == other); }

 private:
  mgp_list *ptr_;
};

/// \brief Wrapper class for \ref mgp_map.
class Map {
 private:
  friend class Value;
  friend class Record;

 public:
  explicit Map(mgp_map *ptr) : ptr_(ptr) {}

  Map &operator=(const Map &other) = delete;
  Map &operator=(Map &&other) = delete;

  size_t size() const { return mgp::map_size(ptr_); }
  bool empty() const { return size() == 0; }

  class Iterator {
   public:
    friend class Map;

    explicit Iterator(mgp_map_items_iterator *map_items_iterator);
    ~Iterator();
    Iterator &operator++();
    bool operator==(Iterator other) const;
    bool operator!=(Iterator other) const { return !(*this == other); }
    MapItem operator*();
    // iterator traits
    using difference_type = MapItem;
    using value_type = MapItem;
    using pointer = const MapItem *;
    using reference = const MapItem &;
    using iterator_category = std::forward_iterator_tag;

   private:
    mgp_map_items_iterator *map_items_iterator_ = nullptr;
  };

  Iterator begin();
  Iterator end();

  void Insert(std::string_view key, const Value &value);
  void Insert(std::string_view key, Value &&value);

  // void Erase(std::string_view key); // (requires mgp_map_erase in the MGP API)

  // void Clear();

  Value const operator[](std::string_view key) const;
  Value const at(std::string_view key) const;

 private:
  mgp_map *ptr_;
};
/* #endregion */

/* #region Graph elements (Vertex, Edge & Path) */
// value types are underlyingly uint8_t values
/// \brief Wrapper class for \ref mgp_vertex.
class Vertex {
 public:
  friend class Graph;
  friend class Path;
  friend class Value;
  friend class Record;

  explicit Vertex(mgp_vertex *ptr) : ptr_(mgp::vertex_copy(ptr, memory)) {}

  /// \brief Create a Node from a copy of the given \ref mgp_vertex.
  explicit Vertex(const mgp_vertex *const_ptr)
      : Vertex(mgp::vertex_copy(const_cast<mgp_vertex *>(const_ptr), memory)) {}

  Vertex(const Vertex &other);
  Vertex(Vertex &&other);

  Vertex &operator=(const Vertex &other) = delete;
  Vertex &operator=(Vertex &&other) = delete;

  ~Vertex();

  Id id() const { return Id::FromInt(mgp::vertex_get_id(ptr_).as_int); }

  Labels labels() const { return Labels(ptr_); }
  bool HasLabel(std::string_view label) const;

  Properties properties() const { return Properties(mgp::vertex_iter_properties(ptr_, memory)); }
  Value operator[](const std::string_view property_name) const;

  Edges in_edges() const;
  Edges out_edges() const;

  bool operator<(const Vertex &other) const { return id() < other.id(); }

  /// \exception std::runtime_error node property contains value with unknown type
  bool operator==(const Vertex &other) const;
  /// \exception std::runtime_error node property contains value with unknown type
  bool operator!=(const Vertex &other) const { return !(*this == other); }

 private:
  mgp_vertex *ptr_;
};

/// \brief Wrapper class for \ref mgp_edge.
class Edge {
 private:
  friend class Value;
  friend class Path;
  friend class Record;

 public:
  explicit Edge(mgp_edge *ptr) : ptr_(mgp::edge_copy(ptr, memory)) {}

  /// \brief Create a Relationship from a copy of the given \ref mgp_edge.
  explicit Edge(const mgp_edge *const_ptr) : Edge(mgp::edge_copy(const_cast<mgp_edge *>(const_ptr), memory)) {}

  Edge(const Edge &other);
  Edge(Edge &&other);

  Edge &operator=(const Edge &other) = delete;
  Edge &operator=(Edge &&other) = delete;

  ~Edge();

  Id id() const { return Id::FromInt(mgp::edge_get_id(ptr_).as_int); }

  /// \brief Return the ID of the relationship’s source node.
  Vertex from() const { return Vertex(mgp::edge_get_from(ptr_)); }
  /// \brief Return the ID of the relationship’s destination node.
  Vertex to() const { return Vertex(mgp::edge_get_to(ptr_)); }

  std::string_view type() const;

  Properties properties() const { return Properties(mgp::edge_iter_properties(ptr_, memory)); }
  Value operator[](const std::string_view property_name) const;

  /// \exception std::runtime_error relationship property contains value with
  /// unknown type
  bool operator==(const Edge &other) const;
  /// \exception std::runtime_error relationship property contains value with
  /// unknown type
  bool operator!=(const Edge &other) const { return !(*this == other); }

  bool operator<(const Edge &other) const { return id() < other.id(); }

 private:
  mgp_edge *ptr_;
};

/// \brief Wrapper class for \ref mgp_path.
class Path {
 private:
  friend class Value;
  friend class Record;

 public:
  explicit Path(mgp_path *ptr) : ptr_(ptr){};

  /// \brief Create a Path from a copy of the given \ref mg_path.
  explicit Path(const mgp_path *const_ptr) : Path(mgp::path_copy(const_cast<mgp_path *>(const_ptr), memory)) {}

  explicit Path(const Vertex &start_vertex);

  Path(const Path &other);
  Path(Path &&other);

  Path &operator=(const Path &other);
  Path &operator=(Path &&other);

  // ~Path();

  /// Length of the path is number of edges.
  size_t length() const { return mgp::path_size(ptr_); }

  /// \brief Returns the vertex at the given `index`.
  /// \pre `index` should be less than or equal to length of the path.
  Vertex GetVertexAt(size_t index) const;

  /// \brief Returns the edge at the given `index`.
  /// \pre `index` should be less than length of the path.
  Edge GetEdgeAt(size_t index) const;

  void Expand(const Edge &edge);

  /// \exception std::runtime_error path contains elements with unknown value
  bool operator==(const Path &other) const;
  /// \exception std::runtime_error path contains elements with unknown value
  bool operator!=(const Path &other) const { return !(*this == other); }

  bool operator<(const Path &other) const { return length() < other.length(); }

 private:
  mgp_path *ptr_;
};
/* #endregion */

/* #region Temporal types (Date, LocalTime, LocalDateTime, Duration) */

/// \brief Wrapper class for \ref mgp_date.
class Date {
 private:
  friend class Duration;
  friend class Value;
  friend class Record;

 public:
  explicit Date(mgp_date *ptr) : ptr_(mgp::date_copy(ptr, memory)) {}

  /// \brief Create a Date from a copy of the given \ref mgp_date.
  explicit Date(const mgp_date *const_ptr) : Date(mgp::date_copy(const_cast<mgp_date *>(const_ptr), memory)) {}

  explicit Date(std::string_view string) : ptr_(mgp::date_from_string(string.data(), memory)) {}

  Date(int year, int month, int day) {
    mgp_date_parameters *params;
    *params = {.year = year, .month = month, .day = day};
    Date(mgp::date_from_parameters(params, memory));
  }

  explicit Date(const Date &other) : Date(mgp::date_copy(other.ptr_, memory)){};
  explicit Date(Date &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; }

  Date &operator=(const Date &other) = delete;
  Date &operator=(Date &&other) = delete;

  ~Date();

  static Date now() { return Date(mgp::date_now(memory)); }

  int year() const { return mgp::date_get_year(ptr_); }
  int month() const { return mgp::date_get_month(ptr_); }
  int day() const { return mgp::date_get_day(ptr_); }

  int64_t timestamp() const { return mgp::date_timestamp(ptr_); }

  bool operator==(const Date &other) const;
  Date operator+(const Duration &dur) const;
  Date operator-(const Duration &dur) const;
  Duration operator-(const Date &other) const;

  bool operator<(const Date &other) const;

 private:
  mgp_date *ptr_;
};

/// \brief Wrapper class for \ref mgp_local_time.
class LocalTime {
 private:
  friend class Duration;
  friend class Value;
  friend class Record;

 public:
  explicit LocalTime(mgp_local_time *ptr) : ptr_(mgp::local_time_copy(ptr, memory)) {}

  /// \brief Create a LocalTime from a copy of the given \ref mgp_local_time.
  explicit LocalTime(const mgp_local_time *const_ptr)
      : LocalTime(mgp::local_time_copy(const_cast<mgp_local_time *>(const_ptr), memory)) {}

  explicit LocalTime(std::string_view string) : ptr_(mgp::local_time_from_string(string.data(), memory)) {}

  LocalTime(int hour, int minute, int second, int millisecond, int microsecond) {
    mgp_local_time_parameters *params;
    *params = {
        .hour = hour, .minute = minute, .second = second, .millisecond = millisecond, .microsecond = microsecond};
    LocalTime(mgp::local_time_from_parameters(params, memory));
  }

  explicit LocalTime(const LocalTime &other) : LocalTime(mgp::local_time_copy(other.ptr_, memory)) {}
  explicit LocalTime(LocalTime &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; };

  LocalTime &operator=(const LocalTime &other) = delete;
  LocalTime &operator=(LocalTime &&other) = delete;

  ~LocalTime();

  static LocalTime now() { return LocalTime(mgp::local_time_now(memory)); }

  int hour() const { return mgp::local_time_get_hour(ptr_); }
  int minute() const { return mgp::local_time_get_minute(ptr_); }
  int second() const { return mgp::local_time_get_second(ptr_); }
  int millisecond() const { return mgp::local_time_get_millisecond(ptr_); }
  int microsecond() const { return mgp::local_time_get_microsecond(ptr_); }

  int64_t timestamp() const { return mgp::local_time_timestamp(ptr_); }

  bool operator==(const LocalTime &other) const;
  LocalTime operator+(const Duration &dur) const;
  LocalTime operator-(const Duration &dur) const;
  Duration operator-(const LocalTime &other) const;

  bool operator<(const LocalTime &other) const;

 private:
  mgp_local_time *ptr_;
};

/// \brief Wrapper class for \ref mgp_local_date_time.
class LocalDateTime {
 private:
  friend class Duration;
  friend class Value;
  friend class Record;

 public:
  explicit LocalDateTime(mgp_local_date_time *ptr) : ptr_(mgp::local_date_time_copy(ptr, memory)) {}

  /// \brief Create a LocalDateTime from a copy of the given \ref mgp_local_date_time.
  explicit LocalDateTime(const mgp_local_date_time *const_ptr)
      : LocalDateTime(mgp::local_date_time_copy(const_cast<mgp_local_date_time *>(const_ptr), memory)) {}

  explicit LocalDateTime(std::string_view string) : ptr_(mgp::local_date_time_from_string(string.data(), memory)) {}

  LocalDateTime(int year, int month, int day, int hour, int minute, int second, int millisecond, int microsecond) {
    struct mgp_date_parameters *date_params;
    struct mgp_local_time_parameters *local_time_params;
    mgp_local_date_time_parameters *params;
    *date_params = {.year = year, .month = month, .day = day};
    *local_time_params = {
        .hour = hour, .minute = minute, .second = second, .millisecond = millisecond, .microsecond = microsecond};
    *params = {.date_parameters = date_params, .local_time_parameters = local_time_params};
    LocalDateTime(mgp::local_date_time_from_parameters(params, memory));
  }

  explicit LocalDateTime(const LocalDateTime &other) : LocalDateTime(mgp::local_date_time_copy(other.ptr_, memory)){};
  explicit LocalDateTime(LocalDateTime &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; };

  LocalDateTime &operator=(const LocalDateTime &other) = delete;
  LocalDateTime &operator=(LocalDateTime &&other) = delete;

  ~LocalDateTime();

  static LocalDateTime now() { return LocalDateTime(mgp::local_date_time_now(memory)); }

  int year() const { return mgp::local_date_time_get_year(ptr_); }
  int month() const { return mgp::local_date_time_get_month(ptr_); }
  int day() const { return mgp::local_date_time_get_day(ptr_); }
  int hour() const { return mgp::local_date_time_get_hour(ptr_); }
  int minute() const { return mgp::local_date_time_get_minute(ptr_); }
  int second() const { return mgp::local_date_time_get_second(ptr_); }
  int millisecond() const { return mgp::local_date_time_get_millisecond(ptr_); }
  int microsecond() const { return mgp::local_date_time_get_microsecond(ptr_); }

  int64_t timestamp() const { return mgp::local_date_time_timestamp(ptr_); }

  bool operator==(const LocalDateTime &other) const;
  LocalDateTime operator+(const Duration &dur) const;
  LocalDateTime operator-(const Duration &dur) const;
  Duration operator-(const LocalDateTime &other) const;

  bool operator<(const LocalDateTime &other) const;

 private:
  mgp_local_date_time *ptr_;
};

/// \brief Wrapper class for \ref mgp_duration.
class Duration {
 private:
  friend class Date;
  friend class LocalTime;
  friend class LocalDateTime;
  friend class Value;
  friend class Record;

 public:
  explicit Duration(mgp_duration *ptr) : ptr_(mgp::duration_copy(ptr, memory)) {}

  /// \brief Create a Duration from a copy of the given \ref mgp_duration.
  explicit Duration(const mgp_duration *const_ptr)
      : Duration(mgp::duration_copy(const_cast<mgp_duration *>(const_ptr), memory)) {}

  explicit Duration(std::string_view string) : ptr_(mgp::duration_from_string(string.data(), memory)) {}

  explicit Duration(int64_t microseconds) : ptr_(mgp::duration_from_microseconds(microseconds, memory)) {}

  Duration(double day, double hour, double minute, double second, double millisecond, double microsecond) {
    mgp_duration_parameters *params;
    *params = {.day = day,
               .hour = hour,
               .minute = minute,
               .second = second,
               .millisecond = millisecond,
               .microsecond = microsecond};
    Duration(mgp::duration_from_parameters(params, memory));
  }

  explicit Duration(const Duration &other) : Duration(mgp::duration_copy(other.ptr_, memory)) {}
  explicit Duration(Duration &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; };

  Duration &operator=(const Duration &other) = delete;
  Duration &operator=(Duration &&other) = delete;

  ~Duration();

  int64_t microseconds() const { return mgp::duration_get_microseconds(ptr_); }

  bool operator==(const Duration &other) const;
  Duration operator+(const Duration &other) const { return Duration(mgp::duration_add(ptr_, other.ptr_, memory)); }
  Duration operator-(const Duration &other) const { return Duration(mgp::duration_sub(ptr_, other.ptr_, memory)); }
  Duration operator-() const { return Duration(mgp::duration_neg(ptr_, memory)); }

  bool operator<(const Duration &other) const;

 private:
  mgp_duration *ptr_;
};
/* #endregion */

/* #endregion */

/* #region Value */
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
  Date,
  LocalTime,
  LocalDateTime,
  Duration,
};

/// Wrapper class for \ref mgp_value
class Value {
 public:
  friend class List;
  friend class Map;
  friend class Date;
  friend class LocalTime;
  friend class LocalDateTime;
  friend class Duration;
  friend class Record;

  explicit Value(mgp_value *ptr_) : ptr_(ptr_){};

  // Primitive type constructors:
  explicit Value(bool value) : Value(mgp::value_make_bool(value, memory)){};
  explicit Value(int value) : Value(mgp::value_make_int(value, memory)){};
  explicit Value(int64_t value) : Value(mgp::value_make_int(value, memory)){};
  explicit Value(double value) : Value(mgp::value_make_double(value, memory)){};

  // String constructors:
  explicit Value(const std::string_view value) : Value(mgp::value_make_string(value.data(), memory)){};
  explicit Value(const char *value) : Value(mgp::value_make_string(value, memory)){};

  // Container constructors:
  explicit Value(List &&list) {
    Value(mgp::value_make_list(list.ptr_));
    delete &list;
    list.ptr_ = nullptr;
  }

  explicit Value(Map &&map) {
    Value(mgp::value_make_map(map.ptr_));
    delete &map;
    map.ptr_ = nullptr;
  }

  // Temporal type constructors:
  explicit Value(Date &&date) {
    Value(mgp::value_make_date(date.ptr_));
    delete &date;
    date.ptr_ = nullptr;
  }
  explicit Value(LocalTime &&local_time) {
    Value(mgp::value_make_local_time(local_time.ptr_));
    delete &local_time;
    local_time.ptr_ = nullptr;
  }
  explicit Value(LocalDateTime &&local_date_time) {
    Value(mgp::value_make_local_date_time(local_date_time.ptr_));
    delete &local_date_time;
    local_date_time.ptr_ = nullptr;
  }
  explicit Value(Duration &&duration) {
    Value(mgp::value_make_duration(duration.ptr_));
    delete &duration;
    duration.ptr_ = nullptr;
  }

  /// \brief Constructs a vertex value and takes ownership of the given `vertex`.
  /// \note The behavior of accessing the `vertex` after performing this operation is undefined.
  explicit Value(Vertex &&vertex) {
    Value(mgp::value_make_vertex(const_cast<mgp_vertex *>(vertex.ptr_)));
    delete &vertex;
    vertex.ptr_ = nullptr;
  };

  /// \brief Constructs an edge value and takes ownership of the given `edge`.
  /// \note The behavior of accessing the `edge` after performing this operation is undefined.
  explicit Value(Edge &&edge) {
    Value(mgp::value_make_edge(edge.ptr_));
    delete &edge;
    edge.ptr_ = nullptr;
  };

  /// \brief Constructs a path value and takes ownership of the given `path`.
  /// \note The behavior of accessing the `path` after performing this operation is undefined.
  explicit Value(Path &&path) {
    Value(mgp::value_make_path(path.ptr_));
    delete &path;
    path.ptr_ = nullptr;
  };

  ~Value();

  const mgp_value *ptr() const { return ptr_; }

  /// \exception std::runtime_error the value type is unknown
  ValueType type() const;

  /// \pre value type is Type::Bool
  bool ValueBool() const;
  /// \pre value type is Type::Int
  int64_t ValueInt() const;
  /// \pre value type is Type::Double
  double ValueDouble() const;
  /// \pre value type is Type::Numeric
  double ValueNumeric() const;
  /// \pre value type is Type::String
  std::string_view ValueString() const;
  /// \pre value type is Type::List
  const List ValueList() const;
  /// \pre value type is Type::Map
  const Map ValueMap() const;
  /// \pre value type is Type::Node
  const Vertex ValueVertex() const;
  /// \pre value type is Type::Edge
  const Edge ValueEdge() const;
  /// \pre value type is Type::Path
  const Path ValuePath() const;
  /// \pre value type is Type::Date
  const Date ValueDate() const;
  /// \pre value type is Type::LocalTime
  const LocalTime ValueLocalTime() const;
  /// \pre value type is Type::LocalDateTime
  const LocalDateTime ValueLocalDateTime() const;
  /// \pre value type is Type::Duration
  const Duration ValueDuration() const;

  bool IsNull() const;
  bool IsBool() const;
  bool IsInt() const;
  bool IsDouble() const;
  bool IsNumeric() const;
  bool IsString() const;
  bool IsList() const;
  bool IsMap() const;
  bool IsVertex() const;
  bool IsEdge() const;
  bool IsPath() const;
  bool IsDate() const;
  bool IsLocalTime() const;
  bool IsLocalDateTime() const;
  bool IsDuration() const;

  /// \exception std::runtime_error the value type is unknown
  bool operator==(const Value &other) const;
  /// \exception std::runtime_error the value type is unknown
  bool operator!=(const Value &other) const { return !(*this == other); }

 private:
  mgp_value *ptr_;
};

/// Key-value pair representing Map items
struct MapItem {
  const std::string_view key;
  const Value value;

  bool operator==(MapItem &other) const;
  bool operator!=(MapItem &other) const;

  bool operator<(const MapItem &other) const { return key < other.key; }
};

/* #endregion */

/* #region Record */

class Record {
 public:
  explicit Record(mgp_result_record *record) : record_(record){};

  void Insert(const char *field_name, std::int64_t value);
  void Insert(const char *field_name, double value);
  void Insert(const char *field_name, const char *value);
  void Insert(const char *field_name, std::string_view value);
  // TODO (convert STL vector to MGP):
  // void Insert(const char *field_name, const std::vector<Value> &list);
  // TODO (convert STL map to MGP):
  // void Insert(const char *field_name, const std::map<std::to_string, Value> &map);
  void Insert(const char *field_name, const Vertex &vertex);
  void Insert(const char *field_name, const Edge &edge);
  void Insert(const char *field_name, const Path &path);
  void Insert(const char *field_name, const Date &date);
  void Insert(const char *field_name, const LocalTime &local_time);
  void Insert(const char *field_name, const LocalDateTime &local_date_time);
  void Insert(const char *field_name, const Duration &duration);

 private:
  void Insert(const char *field_name, const Value &&value);

  mgp_result_record *record_;
};

class RecordFactory {
 public:
  explicit RecordFactory(mgp_result *result) : result_(result){};
  RecordFactory(RecordFactory const &) = delete;

  const mage::Record NewRecord() const;

  void operator=(RecordFactory const &) = delete;

 private:
  mgp_result *result_;
};
/* #endregion */

namespace util {
inline bool ValuesEqual(mgp_value *value1, mgp_value *value2);

inline bool VerticesEqual(mgp_vertex *node1, mgp_vertex *node2) {
  // In query module contexts, vertices with the same ID are considered identical
  if (node1 == node2) {
    return true;
  }
  if (mgp::vertex_get_id(node1).as_int != mgp::vertex_get_id(node2).as_int) {
    return false;
  }
  return true;
}

inline bool EdgesEqual(mgp_edge *edge1, mgp_edge *edge2) {
  // In query module contexts, edges with the same ID are considered identical
  if (edge1 == edge2) {
    return true;
  }
  if (mgp::edge_get_id(edge1).as_int != mgp::edge_get_id(edge2).as_int) {
    return false;
  }
  return true;
}

inline bool PathsEqual(mgp_path *path1, mgp_path *path2) {
  // In query module contexts, paths are considered identical if all their elements are pairwise also identical
  if (path1 == path2) {
    return true;
  }
  if (mgp::path_size(path1) != mgp::path_size(path2)) {
    return false;
  }
  const auto path_size = mgp::path_size(path1);
  for (size_t i = 0; i < path_size; ++i) {
    if (!util::VerticesEqual(mgp::path_vertex_at(path1, i), mgp::path_vertex_at(path2, i))) {
      return false;
    }
    if (!util::EdgesEqual(mgp::path_edge_at(path1, i), mgp::path_edge_at(path2, i))) {
      return false;
    }
  }
  return util::VerticesEqual(mgp::path_vertex_at(path1, path_size), mgp::path_vertex_at(path2, path_size));
}

inline bool ListsEqual(mgp_list *list1, mgp_list *list2) {
  if (list1 == list2) {
    return true;
  }
  if (mgp::list_size(list1) != mgp::list_size(list2)) {
    return false;
  }
  const size_t len = mgp::list_size(list1);
  for (size_t i = 0; i < len; ++i) {
    if (!util::ValuesEqual(mgp::list_at(list1, i), mgp::list_at(list2, i))) {
      return false;
    }
  }
  return true;
}

inline bool MapsEqual(mgp_map *map1, mgp_map *map2) {
  if (map1 == map2) {
    return true;
  }
  if (mgp::map_size(map1) != mgp::map_size(map2)) {
    return false;
  }
  auto items_it = mgp::map_iter_items(map1, memory);
  for (auto item = mgp::map_items_iterator_get(items_it); item; item = mgp::map_items_iterator_next(items_it)) {
    if (mgp::map_item_key(item) == mgp::map_item_key(item)) {
      return false;
    }
    if (!util::ValuesEqual(mgp::map_item_value(item), mgp::map_item_value(item))) {
      return false;
    }
  }
  mgp::map_items_iterator_destroy(items_it);
  return true;
}

inline bool DatesEqual(mgp_date *date1, mgp_date *date2) { return mgp::date_equal(date1, date2); }

inline bool LocalTimesEqual(mgp_local_time *local_time1, mgp_local_time *local_time2) {
  return mgp::local_time_equal(local_time1, local_time2);
}

inline bool LocalDateTimesEqual(mgp_local_date_time *local_date_time1, mgp_local_date_time *local_date_time2) {
  return mgp::local_date_time_equal(local_date_time1, local_date_time2);
}

inline bool DurationsEqual(mgp_duration *duration1, mgp_duration *duration2) {
  return mgp::duration_equal(duration1, duration2);
}

inline bool ValuesEqual(mgp_value *value1, mgp_value *value2) {
  if (value1 == value2) {
    return true;
  }
  if (mgp::value_get_type(value1) != mgp::value_get_type(value2)) {
    return false;
  }
  switch (mgp::value_get_type(value1)) {
    case MGP_VALUE_TYPE_NULL:
      return true;
    case MGP_VALUE_TYPE_BOOL:
      return mgp::value_get_bool(value1) == mgp::value_get_bool(value2);
    case MGP_VALUE_TYPE_INT:
      return mgp::value_get_int(value1) == mgp::value_get_int(value2);
    case MGP_VALUE_TYPE_DOUBLE:
      return mgp::value_get_double(value1) == mgp::value_get_double(value2);
    case MGP_VALUE_TYPE_STRING:
      return std::string_view(mgp::value_get_string(value1)) == std::string_view(mgp::value_get_string(value2));
    case MGP_VALUE_TYPE_LIST:
      return util::ListsEqual(mgp::value_get_list(value1), mgp::value_get_list(value2));
    case MGP_VALUE_TYPE_MAP:
      return util::MapsEqual(mgp::value_get_map(value1), mgp::value_get_map(value2));
      break;
    case MGP_VALUE_TYPE_VERTEX:
      return util::VerticesEqual(mgp::value_get_vertex(value1), mgp::value_get_vertex(value2));
    case MGP_VALUE_TYPE_EDGE:
      return util::EdgesEqual(mgp::value_get_edge(value1), mgp::value_get_edge(value2));
    case MGP_VALUE_TYPE_PATH:
      return util::PathsEqual(mgp::value_get_path(value1), mgp::value_get_path(value2));
    case MGP_VALUE_TYPE_DATE:
      return util::DatesEqual(mgp::value_get_date(value1), mgp::value_get_date(value2));
    case MGP_VALUE_TYPE_LOCAL_TIME:
      return util::LocalTimesEqual(mgp::value_get_local_time(value1), mgp::value_get_local_time(value2));
    case MGP_VALUE_TYPE_LOCAL_DATE_TIME:
      return util::LocalDateTimesEqual(mgp::value_get_local_date_time(value1), mgp::value_get_local_date_time(value2));
    case MGP_VALUE_TYPE_DURATION:
      return util::DurationsEqual(mgp::value_get_duration(value1), mgp::value_get_duration(value2));
  }
  throw ValueException("Invalid value; does not match any Memgraph type.");
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
    case MGP_VALUE_TYPE_DATE:
      return ValueType::Date;
    case MGP_VALUE_TYPE_LOCAL_TIME:
      return ValueType::LocalTime;
    case MGP_VALUE_TYPE_LOCAL_DATE_TIME:
      return ValueType::LocalDateTime;
    case MGP_VALUE_TYPE_DURATION:
      return ValueType::Duration;
    default:
      break;
  }
  std::cout << "type: " << type << "\n";
  throw ValueException("Unknown type error!");
}
}  // namespace util

/* #region Graph (Id, Graph, Vertices, GraphEdges, Edges, Properties & Labels) */

// Graph:

int64_t Graph::order() const {
  int64_t i = 0;
  for (const auto &v : vertices()) {
    i++;
  }
  return i;
}

int64_t Graph::size() const {
  int64_t i = 0;
  for (const auto &_ : edges()) {
    i++;
  }
  return i;
}

inline GraphVertices Graph::vertices() const {
  auto vertices_it = mgp::graph_iter_vertices(graph_, memory);
  if (vertices_it == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }
  return GraphVertices(vertices_it);
}

inline GraphEdges Graph::edges() const { return GraphEdges(graph_); }

inline Vertex Graph::GetVertexById(const Id vertex_id) const {
  auto vertex = mgp::graph_get_vertex_by_id(graph_, mgp_vertex_id{.as_int = vertex_id.AsInt()}, memory);
  return Vertex(vertex);
}

bool Graph::Contains(const Id vertex_id) const { return GetVertexById(vertex_id).ptr_ != nullptr; }
bool Graph::Contains(const Vertex &vertex) const { return Contains(vertex.id()); }
bool Graph::Contains(const Edge &edge) const {
  for (const auto &graph_edge : edges()) {
    if (edge == graph_edge) return true;
  }
  return false;
}

// Vertices:

inline Vertices::Iterator::Iterator(mgp_vertices_iterator *vertices_iterator) : vertices_iterator_(vertices_iterator) {
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

inline Vertex Vertices::Iterator::operator*() {
  if (vertices_iterator_ == nullptr) return Vertex((const mgp_vertex *)nullptr);

  auto vertex = Vertex(mgp::vertices_iterator_get(vertices_iterator_));
  return vertex;
}

inline Vertices::Iterator Vertices::begin() { return Iterator(vertices_iterator_); }

inline Vertices::Iterator Vertices::end() { return Iterator(nullptr); }

// GraphEdges:

inline GraphEdges::Iterator::Iterator(mgp_vertices_iterator *vertices_iterator)
    : vertices_iterator_(vertices_iterator) {
  // Positions the iterator over the first existing edge

  if (vertices_iterator_ == nullptr) return;

  // Go through the adjacent vertices of each graph node
  for (auto vertex = mgp::vertices_iterator_get(vertices_iterator_); vertex;
       vertex = mgp::vertices_iterator_next(vertices_iterator_)) {
    // Check if vertex exists
    if (vertex == nullptr) {
      mgp::vertices_iterator_destroy(vertices_iterator_);
      vertices_iterator_ = nullptr;
      return;
    }

    // Check if vertex has out-edges
    out_edges_iterator_ = mgp::vertex_iter_out_edges(vertex, memory);
    auto edge = mgp::edges_iterator_get(out_edges_iterator_);
    if (edge != nullptr) return;

    mgp::edges_iterator_destroy(out_edges_iterator_);
    out_edges_iterator_ = nullptr;
  }
}

inline GraphEdges::Iterator::~Iterator() {
  if (vertices_iterator_ != nullptr) {
    mgp::vertices_iterator_destroy(vertices_iterator_);
  }
  if (out_edges_iterator_ != nullptr) {
    mgp::edges_iterator_destroy(out_edges_iterator_);
  }
}

inline GraphEdges::Iterator &GraphEdges::Iterator::operator++() {
  // Moves the iterator onto the next existing edge

  // 1. Check if the current vertex has remaining edges to iterate over

  auto edge = mgp::edges_iterator_get(out_edges_iterator_);
  if (edge != nullptr) return *this;

  mgp::edges_iterator_destroy(out_edges_iterator_);
  out_edges_iterator_ = nullptr;

  // 2. Move onto the next vertices

  if (vertices_iterator_ != nullptr) {
    for (auto vertex = mgp::vertices_iterator_get(vertices_iterator_); vertex;
         vertex = mgp::vertices_iterator_next(vertices_iterator_)) {
      // Check if vertex exists
      if (vertex == nullptr) {
        mgp::vertices_iterator_destroy(vertices_iterator_);
        vertices_iterator_ = nullptr;
        return *this;
      }

      // Check if vertex has out-edges
      out_edges_iterator_ = mgp::vertex_iter_out_edges(vertex, memory);
      auto edge = mgp::edges_iterator_get(out_edges_iterator_);
      if (edge != nullptr) return *this;

      mgp::edges_iterator_destroy(out_edges_iterator_);
      out_edges_iterator_ = nullptr;
    }

    mgp::vertices_iterator_destroy(vertices_iterator_);
    vertices_iterator_ = nullptr;
  }
  return *this;
}

inline bool GraphEdges::Iterator::operator==(Iterator other) const {
  if (out_edges_iterator_ == nullptr && other.out_edges_iterator_ == nullptr) {
    return true;
  }
  if (out_edges_iterator_ == nullptr || other.out_edges_iterator_ == nullptr) {
    return false;
  }
  return mgp::edge_equal(mgp::edges_iterator_get(out_edges_iterator_),
                         mgp::edges_iterator_get(other.out_edges_iterator_)) &&
         index_ == other.index_;
}

inline Edge GraphEdges::Iterator::operator*() {
  if (out_edges_iterator_ != nullptr) {
    return Edge(mgp::edges_iterator_get(out_edges_iterator_));
  }

  return Edge((mgp_edge *)nullptr);
}

inline GraphEdges::Iterator GraphEdges::begin() { return Iterator(mgp::graph_iter_vertices(graph_, memory)); }

inline GraphEdges::Iterator GraphEdges::end() { return Iterator(nullptr); }

// Edges:

inline Edges::Iterator::Iterator(mgp_edges_iterator *edges_iterator) : edges_iterator_(edges_iterator) {
  if (edges_iterator_ == nullptr) return;
  if (mgp::edges_iterator_get(edges_iterator_) == nullptr) {
    mgp::edges_iterator_destroy(edges_iterator_);
    edges_iterator_ = nullptr;
  }
}

inline Edges::Iterator::~Iterator() {
  std::cout << "destroy\n";
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

inline Edge Edges::Iterator::operator*() {
  if (edges_iterator_ == nullptr) return Edge((mgp_edge *)nullptr);

  auto edge = Edge(mgp::edges_iterator_get(edges_iterator_));
  return edge;
}

inline Edges::Iterator Edges::begin() { return Iterator(edges_iterator_); }

inline Edges::Iterator Edges::end() { return Iterator(nullptr); }

// Properties:

inline Properties::Properties(mgp_properties_iterator *properties_iterator) {
  for (auto property = mgp::properties_iterator_get(properties_iterator); property;
       property = mgp::properties_iterator_next(properties_iterator)) {
    auto value = Value(property->value);
    property_map_.emplace(property->name, value);
  }
  mgp::properties_iterator_destroy(properties_iterator);
}

inline Value Properties::operator[](const std::string_view key) const { return property_map_.at(key); }

inline bool Properties::operator==(const Properties &other) const { return property_map_ == other.property_map_; }

// Labels:

inline std::string_view Labels::Iterator::operator*() const { return (*iterable_)[index_]; }

inline std::string_view Labels::operator[](size_t index) const { return mgp::vertex_label_at(vertex_ptr_, index).name; }
/* #endregion */

/* #region Types */

/* #region Containers (List, Map) */

// List:

inline Value List::Iterator::operator*() const { return (*iterable_)[index_]; }

inline const Value List::operator[](size_t index) const { return Value(mgp::list_at(ptr_, index)); }

inline void List::Append(const Value &value) { mgp::list_append(ptr_, value.ptr_); }

inline void List::Append(Value &&value) {
  mgp::list_append(ptr_, value.ptr_);
  value.ptr_ = nullptr;
}

inline void List::AppendExtend(Value &value) { mgp::list_append_extend(ptr_, value.ptr_); }

inline bool List::operator==(const List &other) const { return util::ListsEqual(ptr_, other.ptr_); }

inline bool MapItem::operator==(MapItem &other) const { return key == other.key && value == other.value; }
inline bool MapItem::operator!=(MapItem &other) const { return !(*this == other); }

// Map:

inline Map::Iterator::Iterator(mgp_map_items_iterator *map_items_iterator) : map_items_iterator_(map_items_iterator) {
  if (map_items_iterator_ == nullptr) return;
  if (mgp::map_items_iterator_get(map_items_iterator_) == nullptr) {
    mgp::map_items_iterator_destroy(map_items_iterator_);
    map_items_iterator_ = nullptr;
  }
}

inline Map::Iterator::~Iterator() {
  if (map_items_iterator_ != nullptr) {
    mgp::map_items_iterator_destroy(map_items_iterator_);
  }
}

inline Map::Iterator &Map::Iterator::operator++() {
  if (map_items_iterator_ != nullptr) {
    auto next = mgp::map_items_iterator_next(map_items_iterator_);

    if (next == nullptr) {
      mgp::map_items_iterator_destroy(map_items_iterator_);
      map_items_iterator_ = nullptr;
      return *this;
    }
  }
  return *this;
}

inline bool Map::Iterator::operator==(Iterator other) const {
  if (map_items_iterator_ == nullptr && other.map_items_iterator_ == nullptr) {
    return true;
  }
  if (map_items_iterator_ == nullptr || other.map_items_iterator_ == nullptr) {
    return false;
  }
  return mgp::map_items_iterator_get(map_items_iterator_) == mgp::map_items_iterator_get(other.map_items_iterator_);
}

inline MapItem Map::Iterator::operator*() {
  if (map_items_iterator_ == nullptr) {
    throw ValueException("Empty map item!");
  }

  auto raw_map_item = mgp::map_items_iterator_get(map_items_iterator_);

  auto map_key = mgp::map_item_key(raw_map_item);
  auto map_value = Value(mgp::map_item_value(raw_map_item));

  return MapItem{.key = map_key, .value = map_value};
}

inline Map::Iterator Map::begin() { return Iterator(mgp::map_iter_items(ptr_, memory)); }

inline Map::Iterator Map::end() { return Iterator(nullptr); }

inline void Map::Insert(std::string_view key, const Value &value) { mgp::map_insert(ptr_, key.data(), value.ptr_); }

inline void Map::Insert(std::string_view key, Value &&value) {
  mgp::map_insert(ptr_, key.data(), value.ptr_);
  value.ptr_ = nullptr;
}

// inline void Map::Clear() {
//   mgp::map_destroy(ptr_);
//   ptr_ = mgp::map_make_empty(memory);
// }

inline const Value Map::operator[](std::string_view key) const { return Value(mgp::map_at(ptr_, key.data())); }

inline const Value Map::at(std::string_view key) const { return Value(mgp::map_at(ptr_, key.data())); }
/* #endregion */

/* #region Graph elements (Vertex, Edge & Path) */

// Vertex:

inline Vertex::Vertex(const Vertex &other) : Vertex(mgp::vertex_copy(other.ptr_, memory)) {}

inline Vertex::Vertex(Vertex &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; }

inline Vertex::~Vertex() {
  if (ptr_ != nullptr) {
    mgp::vertex_destroy(ptr_);
  }
}

inline bool Vertex::HasLabel(std::string_view label) const {
  for (const auto vertex_label : labels()) {
    if (label == vertex_label) {
      return true;
    }
  }
  return false;
}

inline Value Vertex::operator[](const std::string_view property_name) const { return properties()[property_name]; }

inline Edges Vertex::in_edges() const {
  auto edge_iterator = mgp::vertex_iter_in_edges(ptr_, memory);
  if (edge_iterator == nullptr) {
    throw NotEnoughMemoryException();
  }
  return Edges(edge_iterator);
}

inline Edges Vertex::out_edges() const {
  auto edge_iterator = mgp::vertex_iter_out_edges(ptr_, memory);
  if (edge_iterator == nullptr) {
    throw NotEnoughMemoryException();
  }
  return Edges(edge_iterator);
}

inline bool Vertex::operator==(const Vertex &other) const { return util::VerticesEqual(ptr_, other.ptr_); }

// Edge:

inline Edge::Edge(const Edge &other) : Edge(mgp::edge_copy(other.ptr_, memory)) {}

inline Edge::Edge(Edge &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; }

inline Edge::~Edge() {
  if (ptr_ != nullptr) {
    mgp::edge_destroy(ptr_);
  }
}

inline std::string_view Edge::type() const { return mgp::edge_get_type(ptr_).name; }

inline Value Edge::operator[](const std::string_view property_name) const { return properties()[property_name]; }

inline bool Edge::operator==(const Edge &other) const { return util::EdgesEqual(ptr_, other.ptr_); }

// Path:

inline Path::Path(const Path &other) : ptr_(mgp::path_copy(other.ptr_, memory)) {}

inline Path::Path(Path &&other) : ptr_(other.ptr_) { other.ptr_ = nullptr; }

inline Path::Path(const Vertex &start_vertex) : ptr_(mgp::path_make_with_start(start_vertex.ptr_, memory)) {}

// inline Path::~Path() {
//   if (ptr_ != nullptr) {
//     mgp::path_destroy(ptr_);
//   }
// }

inline Vertex Path::GetVertexAt(size_t index) const {
  auto vertex_ptr = mgp::path_vertex_at(ptr_, index);
  if (vertex_ptr == nullptr) {
    throw IndexException("Index value out of bounds.");
  }
  return Vertex(vertex_ptr);
}

inline Edge Path::GetEdgeAt(size_t index) const {
  auto edge_ptr = mgp::path_edge_at(ptr_, index);
  if (edge_ptr == nullptr) {
    throw IndexException("Index value out of bounds.");
  }
  return Edge(edge_ptr);
}

inline void Path::Expand(const Edge &edge) { mgp::path_expand(ptr_, edge.ptr_); }

inline bool Path::operator==(const Path &other) const { return util::PathsEqual(ptr_, other.ptr_); }
/* #endregion */

/* #region Temporal types (Date, LocalTime, LocalDateTime, Duration) */

// Date:

inline Date::~Date() {
  if (ptr_ != nullptr) {
    mgp::date_destroy(ptr_);
  }
}

inline bool Date::operator==(const Date &other) const { return util::DatesEqual(ptr_, other.ptr_); }

inline Date Date::operator+(const Duration &dur) const { return Date(mgp::date_add_duration(ptr_, dur.ptr_, memory)); }

inline Date Date::operator-(const Duration &dur) const { return Date(mgp::date_sub_duration(ptr_, dur.ptr_, memory)); }

inline Duration Date::operator-(const Date &other) const { return Duration(mgp::date_diff(ptr_, other.ptr_, memory)); }

inline bool Date::operator<(const Date &other) const {
  auto difference = mgp::date_diff(ptr_, other.ptr_, memory);
  return (mgp::duration_get_microseconds(difference) < 0);
}

// LocalTime:

inline LocalTime::~LocalTime() {
  if (ptr_ != nullptr) {
    mgp::local_time_destroy(ptr_);
  }
}

inline bool LocalTime::operator==(const LocalTime &other) const { return util::LocalTimesEqual(ptr_, other.ptr_); }

inline LocalTime LocalTime::operator+(const Duration &dur) const {
  return LocalTime(mgp::local_time_add_duration(ptr_, dur.ptr_, memory));
}

inline LocalTime LocalTime::operator-(const Duration &dur) const {
  return LocalTime(mgp::local_time_sub_duration(ptr_, dur.ptr_, memory));
}

inline Duration LocalTime::operator-(const LocalTime &other) const {
  return Duration(mgp::local_time_diff(ptr_, other.ptr_, memory));
}

inline bool LocalTime::operator<(const LocalTime &other) const {
  auto difference = mgp::local_time_diff(ptr_, other.ptr_, memory);
  return (mgp::duration_get_microseconds(difference) < 0);
}

// LocalDateTime:

inline LocalDateTime::~LocalDateTime() {
  if (ptr_ != nullptr) {
    mgp::local_date_time_destroy(ptr_);
  }
}

inline bool LocalDateTime::operator==(const LocalDateTime &other) const {
  return util::LocalDateTimesEqual(ptr_, other.ptr_);
}

inline LocalDateTime LocalDateTime::operator+(const Duration &dur) const {
  return LocalDateTime(mgp::local_date_time_add_duration(ptr_, dur.ptr_, memory));
}

inline LocalDateTime LocalDateTime::operator-(const Duration &dur) const {
  return LocalDateTime(mgp::local_date_time_sub_duration(ptr_, dur.ptr_, memory));
}

inline Duration LocalDateTime::operator-(const LocalDateTime &other) const {
  return Duration(mgp::local_date_time_diff(ptr_, other.ptr_, memory));
}

inline bool LocalDateTime::operator<(const LocalDateTime &other) const {
  auto difference = mgp::local_date_time_diff(ptr_, other.ptr_, memory);
  return (mgp::duration_get_microseconds(difference) < 0);
}

// Duration:

inline Duration::~Duration() {
  if (ptr_ != nullptr) {
    mgp::duration_destroy(ptr_);
  }
}

inline bool Duration::operator==(const Duration &other) const { return util::DurationsEqual(ptr_, other.ptr_); }

inline bool Duration::operator<(const Duration &other) const {
  auto difference = mgp::duration_sub(ptr_, other.ptr_, memory);
  return (mgp::duration_get_microseconds(difference) < 0);
}
/* #endregion */

/* #endregion */

/* #region Value */
inline Value::~Value() {
  if (ptr_ != nullptr) {
    mgp::value_destroy(ptr_);
  }
}

inline ValueType Value::type() const { return util::ConvertType(mgp::value_get_type(ptr_)); }

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
  std::cout << "ds\n";
  return mgp::value_get_int(ptr_);
}

inline double Value::ValueDouble() const {
  if (type() != ValueType::Double) {
    throw ValueException("Type of value is wrong: expected Double.");
  }
  return mgp::value_get_double(ptr_);
}

inline double Value::ValueNumeric() const {
  if (type() != ValueType::Int || type() != ValueType::Double) {
    throw ValueException("Type of value is wrong: expected Int or Double.");
  }
  if (type() == ValueType::Int) {
    return static_cast<double>(mgp::value_get_int(ptr_));
  }
  return mgp::value_get_double(ptr_);
}

inline std::string_view Value::ValueString() const {
  if (type() != ValueType::String) {
    throw ValueException("Type of value is wrong: expected String.");
  }
  return mgp::value_get_string(ptr_);
}

inline const List Value::ValueList() const {
  if (type() != ValueType::List) {
    throw ValueException("Type of value is wrong: expected List.");
  }
  return List(mgp::value_get_list(ptr_));
}

inline const Map Value::ValueMap() const {
  if (type() != ValueType::Map) {
    throw ValueException("Type of value is wrong: expected Map.");
  }
  return Map(mgp::value_get_map(ptr_));
}

inline const Vertex Value::ValueVertex() const {
  if (type() != ValueType::Vertex) {
    throw ValueException("Type of value is wrong: expected Vertex.");
  }
  return Vertex(mgp::value_get_vertex(ptr_));
}

inline const Edge Value::ValueEdge() const {
  if (type() != ValueType::Edge) {
    throw ValueException("Type of value is wrong: expected Edge.");
  }
  return Edge(mgp::value_get_edge(ptr_));
}

inline const Path Value::ValuePath() const {
  if (type() != ValueType::Path) {
    throw ValueException("Type of value is wrong: expected Path.");
  }
  return Path(mgp::value_get_path(ptr_));
}

inline const Date Value::ValueDate() const {
  if (type() != ValueType::Date) {
    throw ValueException("Type of value is wrong: expected Date.");
  }
  return Date(mgp::value_get_date(ptr_));
}

inline const LocalTime Value::ValueLocalTime() const {
  if (type() != ValueType::Date) {
    throw ValueException("Type of value is wrong: expected LocalTime.");
  }
  return LocalTime(mgp::value_get_local_time(ptr_));
}

inline const LocalDateTime Value::ValueLocalDateTime() const {
  if (type() != ValueType::LocalDateTime) {
    throw ValueException("Type of value is wrong: expected LocalDateTime.");
  }
  return LocalDateTime(mgp::value_get_local_date_time(ptr_));
}

inline const Duration Value::ValueDuration() const {
  if (type() != ValueType::Duration) {
    throw ValueException("Type of value is wrong: expected Duration.");
  }
  return Duration(mgp::value_get_duration(ptr_));
}

inline bool Value::IsNull() const { return mgp::value_is_null(ptr_); }
inline bool Value::IsBool() const { return mgp::value_is_bool(ptr_); }
inline bool Value::IsInt() const { return mgp::value_is_int(ptr_); }
inline bool Value::IsDouble() const { return mgp::value_is_double(ptr_); }
inline bool Value::IsNumeric() const { return IsInt() || IsDouble(); }
inline bool Value::IsString() const { return mgp::value_is_string(ptr_); }
inline bool Value::IsList() const { return mgp::value_is_list(ptr_); }
inline bool Value::IsMap() const { return mgp::value_is_map(ptr_); }
inline bool Value::IsVertex() const { return mgp::value_is_vertex(ptr_); }
inline bool Value::IsEdge() const { return mgp::value_is_edge(ptr_); }
inline bool Value::IsPath() const { return mgp::value_is_path(ptr_); }
inline bool Value::IsDate() const { return mgp::value_is_date(ptr_); }
inline bool Value::IsLocalTime() const { return mgp::value_is_local_time(ptr_); }
inline bool Value::IsLocalDateTime() const { return mgp::value_is_local_date_time(ptr_); }
inline bool Value::IsDuration() const { return mgp::value_is_duration(ptr_); }

inline bool Value::operator==(const Value &other) const { return util::ValuesEqual(ptr_, other.ptr_); }
/* #endregion */

/* #region Record */
// Record:

inline void Record::Insert(const char *field_name, std::int64_t value) { Insert(field_name, Value(value)); }
inline void Record::Insert(const char *field_name, double value) { Insert(field_name, Value(value)); }
inline void Record::Insert(const char *field_name, const char *value) { Insert(field_name, Value(value)); }
inline void Record::Insert(const char *field_name, std::string_view value) { Insert(field_name, Value(value)); }
// TODO (convert STL vector to MGP):
// inline void Record::Insert(const char *field_name, std::vector<Value> &list) {
//   Insert(field_name, Value(mgp::value_make_list(mgp::list_copy(list.ptr_, memory))));
// }
// TODO (convert STL map to MGP):
// inline void Record::Insert(const char *field_name, std::map<std::string_view, Value> &list) {
//   Insert(field_name, Value(mgp::value_make_map(mgp::map_copy(map.ptr_, memory))));
// }
inline void Record::Insert(const char *field_name, const Vertex &vertex) {
  Insert(field_name, Value(mgp::value_make_vertex(mgp::vertex_copy(vertex.ptr_, memory))));
}
inline void Record::Insert(const char *field_name, const Edge &edge) {
  Insert(field_name, Value(mgp::value_make_edge(mgp::edge_copy(edge.ptr_, memory))));
}
inline void Record::Insert(const char *field_name, const Path &path) {
  Insert(field_name, Value(mgp::value_make_path(mgp::path_copy(path.ptr_, memory))));
}
inline void Record::Insert(const char *field_name, const Date &date) {
  Insert(field_name, Value(mgp::value_make_date(mgp::date_copy(date.ptr_, memory))));
}
inline void Record::Insert(const char *field_name, const LocalTime &local_time) {
  Insert(field_name, Value(mgp::value_make_local_time(mgp::local_time_copy(local_time.ptr_, memory))));
}
inline void Record::Insert(const char *field_name, const LocalDateTime &local_date_time) {
  Insert(field_name, Value(mgp::value_make_local_date_time(mgp::local_date_time_copy(local_date_time.ptr_, memory))));
}
inline void Record::Insert(const char *field_name, const Duration &duration) {
  Insert(field_name, Value(mgp::value_make_duration(mgp::duration_copy(duration.ptr_, memory))));
}
inline void Record::Insert(const char *field_name, const Value &&value) {
  mgp::result_record_insert(record_, field_name, value.ptr_);
}

// RecordFactory:

inline const Record RecordFactory::NewRecord() const {
  auto record = mgp::result_new_record(result_);
  if (record == nullptr) {
    throw NotEnoughMemoryException();
  }
  return Record(record);
}
/* #endregion */

}  // namespace mage

namespace std {
template <>
struct hash<mage::Id> {
  size_t operator()(const mage::Id &x) const { return hash<int64_t>()(x.AsInt()); };
};

template <>
struct hash<mage::Vertex> {
  size_t operator()(const mage::Vertex &x) const { return hash<int64_t>()(x.id().AsInt()); };
};

template <>
struct hash<mage::Edge> {
  size_t operator()(const mage::Edge &x) const { return hash<int64_t>()(x.id().AsInt()); };
};

// TODO how do I hash a Path

template <>
struct hash<mage::Date> {
  size_t operator()(const mage::Date &x) const { return hash<int64_t>()(x.timestamp()); };
};

template <>
struct hash<mage::LocalTime> {
  size_t operator()(const mage::LocalTime &x) const { return hash<int64_t>()(x.timestamp()); };
};

template <>
struct hash<mage::LocalDateTime> {
  size_t operator()(const mage::LocalDateTime &x) const { return hash<int64_t>()(x.timestamp()); };
};

template <>
struct hash<mage::Duration> {
  size_t operator()(const mage::Duration &x) const { return hash<int64_t>()(x.microseconds()); };
};

// TODO how do I hash a Value

template <>
struct hash<mage::MapItem> {
  size_t operator()(const mage::MapItem &x) const { return hash<std::string_view>()(x.key); };
};
}  // namespace std