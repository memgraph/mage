#include <mgp.hpp>


extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
   

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
