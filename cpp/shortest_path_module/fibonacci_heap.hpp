#include <cmath>
#include <functional>
#include <iostream>
#include <stack>
#include <unordered_map>
#include <vector>

/**
 *  usage example in SP's algorithm
 *  the KEY would be the distance to each node, initialized at INF
 *  the DATA would be an identifier for the node itself
 *  for this reason all element's DATA should be different from one another while
 *  it's reasonable to have two element with the same KEY
 *
 *  C++11 or above
 *
 **/

// fibonacci heap
template <typename KEY, typename DATA>
class fibonacci_heap {
 private:
  // doubly linked list template
  template <typename T>
  class doubly_linked_list {
   private:
    T _head;
    size_t _size;

    void insert_after(T prev, T node) {
      if (prev != nullptr) {
        node->left = prev;
        node->right = prev->right;
        prev->right->left = node;
        prev->right = node;
      }
      ++_size;
    }

   public:
    doubly_linked_list() : _head(nullptr), _size(0){};

    T head() { return _head; }

    void remove(T node) {  // remove node from list and clear its pointers
      if (_size == 1) {
        _head = nullptr;
      } else if (node == _head) {
        _head = node->right;
      }
      node->right->left = node->left;
      node->left->right = node->right;
      node->left = node;
      node->right = node;
      --_size;
    }

    void push_back(T node) {  // append the node at the tail of the list
      if (empty())
        insert_after(_head = node, node);
      else
        insert_after(_head->left, node);
    }

    void merge(doubly_linked_list &l) {
      T last_node = _head->left;
      _head->left = l._head->left;
      l._head->left->right = _head;
      l._head->left = last_node;
      last_node->right = l._head;
      _size += l.size();
    }

    void print() {
      T n = _head;
      do {
        std::cout << n->key << " @" << n << std::endl;
        n = n->right;
      } while (n != _head);
    }

    bool empty() { return !_size; }

    ssize_t size() { return _size; }

    void clear() { _head = nullptr, _size = 0; }
  };

  template <typename KEY_NODE, typename DATA_NODE>
  class fibonacci_heap_node {
   public:
    fibonacci_heap_node<KEY_NODE, DATA_NODE> *p, *left, *right;
    doubly_linked_list<fibonacci_heap_node<KEY_NODE, DATA_NODE> *> child_list;
    ssize_t degree;
    bool mark;
    KEY_NODE key;
    DATA_NODE data;

    fibonacci_heap_node() : p(nullptr), left(this), right(this), child_list(), degree(0), mark(false) {}
    // If obj are modified after being inseted something bad could happen
    fibonacci_heap_node(KEY_NODE &k, DATA_NODE &d) : fibonacci_heap_node<KEY_NODE, DATA_NODE>() {
      key = k;
      data = d;
    }

    void clear() {
      p = nullptr;
      left = this;
      right = this;
      child_list = doubly_linked_list<fibonacci_heap_node<KEY_NODE, DATA_NODE> *>();
      degree = 0;
      mark = false;
    }

    bool operator<(const fibonacci_heap_node x) { return key < x.key; }
    bool operator>(const fibonacci_heap_node x) { return key > x.key; }
  };

  const int POOL_SIZE = 1000;
  ssize_t nodes;
  // pool implemented using a stack instead of a queue
  // to reuse nodes as soon as possible (cache friendly?)
  std::stack<fibonacci_heap_node<KEY, DATA> *> pool;
  fibonacci_heap_node<KEY, DATA> *top_node;
  doubly_linked_list<fibonacci_heap_node<KEY, DATA> *> root_list;
  std::unordered_map<DATA, fibonacci_heap_node<KEY, DATA> *> addresses;
  std::function<bool(KEY, KEY)> compare;

  void consolidate() {
    std::vector<fibonacci_heap_node<KEY, DATA> *> pointers(max_degree(), nullptr);
    fibonacci_heap_node<KEY, DATA> *node = top_node, *x, *y;

    for (ssize_t i = 0; i < root_list.size(); ++i) {
      node = (x = node)->right;  // x = node, node = x->right
      ssize_t d = x->degree;
      while (pointers[d]) {
        y = pointers[d];
        if (!compare(x->key, y->key)) std::swap(x, y);
        make_child(y, x);
        pointers[d] = nullptr;
        ++d, --i;
      }
      pointers[d] = x;
    }
    root_list.clear();
    top_node = nullptr;
    for (auto &x : pointers) {
      if (x) {
        root_list.push_back(x);
        if (top_node == nullptr) {
          top_node = x;
        } else if (compare(x->key, top_node->key)) {
          top_node = x;
        }
      }
    }
  }

  void cut(fibonacci_heap_node<KEY, DATA> *x, fibonacci_heap_node<KEY, DATA> *y) {
    y->child_list.remove(x);
    --y->degree;
    root_list.push_back(x);
    x->p = nullptr;
    x->mark = false;
  }

  void cascading_cut(fibonacci_heap_node<KEY, DATA> *y) {
    fibonacci_heap_node<KEY, DATA> *z = y->p;
    while (z != nullptr) {
      if (y->mark == false) {
        y->mark = true;
        z = nullptr;
      } else {
        cut(y, z);
        z = (y = z)->p;
      }
    }
  }

  void make_child(fibonacci_heap_node<KEY, DATA> *y, fibonacci_heap_node<KEY, DATA> *x) {
    root_list.remove(y);
    x->child_list.push_back(y);
    ++x->degree;
    y->p = x;
    y->mark = false;
  }

  void insert(fibonacci_heap_node<KEY, DATA> *node) {
    addresses[node->data] = node;
    root_list.push_back(node);
    if (top_node == nullptr) {
      top_node = node;
    } else if (compare(node->key, top_node->key)) {
      top_node = node;
    }
    ++nodes;
  }

  // upper_bound of number of root nodes in the root lists that will be present after consolidation
  ssize_t max_degree() { return (ssize_t)floor(log((double)nodes) / log((1.0 + sqrt(5.0)) / 2.0)) + 1; }

  void fill_pool() {
    for (ssize_t i = 0; i < POOL_SIZE; i++) {
      pool.push(new fibonacci_heap_node<KEY, DATA>());
    }
  }

  fibonacci_heap_node<KEY, DATA> *get_node(KEY &k, DATA &d) {
    fibonacci_heap_node<KEY, DATA> *x = pool.top();
    pool.pop();
    x->key = k;
    x->data = d;
    if (pool.size() == 0) fill_pool();
    return x;
  }

 public:
  fibonacci_heap(std::function<bool(KEY, KEY)> cmp) : nodes(0), top_node(nullptr), addresses(), compare(cmp) {
    fill_pool();
  }

  bool empty() { return !nodes; }

  void insert(KEY k, DATA d) { insert(get_node(k, d)); }

  std::pair<KEY, DATA> get() { return {top_node->key, top_node->data}; }

  void remove() {
    fibonacci_heap_node<KEY, DATA> *extracted = top_node;
    if (extracted != nullptr) {
      while (extracted->child_list.size()) {
        fibonacci_heap_node<KEY, DATA> *child = extracted->child_list.head()->right;
        extracted->child_list.remove(child);
        child->p = nullptr;
        root_list.push_back(child);
      }
      fibonacci_heap_node<KEY, DATA> *next_node = extracted->right;
      root_list.remove(extracted);
      if (extracted == next_node) {
        top_node = nullptr;
      } else {
        top_node = next_node;
        consolidate();
      }
      --nodes;
      addresses.erase(extracted->data);
      extracted->clear();
      pool.push(extracted);
    }
  }

  void update_key(KEY key, DATA data) {
    if (addresses.count(data) == 0) return;
    fibonacci_heap_node<KEY, DATA> *node = addresses[data];
    if (compare(key, node->key)) {
      node->key = key;
      fibonacci_heap_node<KEY, DATA> *parent = node->p;
      if (parent != nullptr && compare(node->key, parent->key)) {
        cut(node, parent);
        cascading_cut(parent);
      }
      if (compare(node->key, top_node->key)) {
        top_node = node;
      }
    }
  }

  void merge(fibonacci_heap<KEY, DATA> &other) {
    root_list.merge(other.root_list);

    if (top_node == nullptr || (other.top_node != nullptr && compare(other.top_node->key, top_node->key)))
      top_node = other.top_node;

    nodes += other.size();
  }

  ssize_t size() { return nodes; }
};
