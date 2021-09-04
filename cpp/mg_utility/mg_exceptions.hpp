#pragma once

#include <exception>
#include <iostream>

namespace mg_exception {
struct NotEnoughMemoryException : public std::exception {
  const char *what() const throw() { return "Not enough memory!"; }
};

struct UnknownException : public std::exception {
  const char *what() const throw() { return "Unknown exception!"; }
};
struct AllocationException : public std::exception {
  const char *what() const throw() { return "Could not allocate memory!"; }
};
struct InsufficientBufferException : public std::exception {
  const char *what() const throw() { return "Buffer is not sufficient to process procedure!"; }
};
struct OutOfRangeException : public std::exception {
  const char *what() const throw() { return "Index out of range!"; }
};
struct LogicException : public std::exception {
  const char *what() const throw() { return "Logic exception, check the procedure signature!"; }
};
struct NonExistendObjectException : public std::exception {
  const char *what() const throw() { return "Object does not exist!"; }
};
struct InvalidArgumentException : public std::exception {
  const char *what() const throw() { return "Invalid argument!"; }
};

struct InvalidIDException : public std::exception {
  const char *what() const throw() { return "Invalid ID!"; }
};
}  // namespace mg_exception