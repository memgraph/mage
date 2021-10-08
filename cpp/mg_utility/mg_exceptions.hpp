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
struct DeletedObjectException : public std::exception {
  const char *what() const throw() { return "Object is deleted!"; }
};
struct InvalidArgumentException : public std::exception {
  const char *what() const throw() { return "Invalid argument!"; }
};

struct InvalidIDException : public std::exception {
  const char *what() const throw() { return "Invalid ID!"; }
};
struct KeyAlreadyExistsException : public std::exception {
  const char *what() const throw() { return "Key you are trying to set already exists!"; }
};

struct ImmutableObjectException : public std::exception {
  const char *what() const throw() { return "Object you are trying to change is immutable!"; }
};
struct ValueConversionException : public std::exception {
  const char *what() const throw() { return "Error in value conversion!"; }
};
struct SerializationException : public std::exception {
  const char *what() const throw() { return "Error in serialization!"; }
};
}  // namespace mg_exception