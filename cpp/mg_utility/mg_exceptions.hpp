#include <exception>
#include <iostream>

namespace mg_exception {
struct NotEnoughMemoryException : public std::exception {
  const char *what() const throw() { return "Not enough memory!"; }
};

struct InvalidIDException : public std::exception {
  const char *what() const throw() { return "Invalid ID!"; }
};
} // namespace mg_exception