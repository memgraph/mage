#include <iostream>

#include <mgp.hpp>

extern "C" int mgp_init_module(struct mgp_module *module,
                               struct mgp_memory *memory) {
  try {
    std::cout << "before" << "\n";
    std::cout << 10 << "\n";
    std::cout << "after"
              << "\n";
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
