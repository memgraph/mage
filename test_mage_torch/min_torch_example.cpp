#include <python3.10/Python.h>


int main(void) {
  setenv("PYTHONPATH",".",1);
  Py_Initialize();

  PyObject *res = PyImport_ImportModule("test_module.py");

  Py_Finalize();
  
  return 0;
}
