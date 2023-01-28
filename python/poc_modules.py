import ast
from mage.test_module.test_functions import test_function
from mage.test_module.test_functions_dir.test_functions import test_function_minus
import sys
import subprocess
import networkx


modules = set()
no_removals = [
    "collections",
    "abc",
    "numpy",
    "sys",
    "networkx",
    "scipy",
    "mgp_networkx",
]


def visit_Import(node):
    for name in node.names:
        if name.name not in no_removals:
            # modules.add(name.name.split(".")[0])
            modules.add(name.name)


def visit_ImportFrom(node):
    # if node.module is missing it's a "from . import ..." statement
    # if level > 0 it's a "from .submodule import ..." statement
    if node.module is not None and node.level == 0:
        # modules.add(node.module)
        mod_name = node.module.split(".")[0]
        if mod_name not in no_removals:
            modules.add(mod_name)
        # modules.add(node.module.split(".")[0])


def poc_pip(uninstall: str, install: str):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", uninstall])
    subprocess.check_call([sys.executable, "-m", "pip", "install", install])


def remove_custom_module_and_packages_from_cache(modules):
    # Can we really delete all keys?
    poc_pip("networkx==2.6.2", "networkx==2.6.3")
    to_del = []
    for key in sys.modules.keys():
        if key in modules:
            print(key)
            to_del.append(key)
    for key in to_del:
        del sys.modules[key]
    import from_import_test

    poc_pip("networkx==2.6.3", "networkx==2.6.2")


if __name__ == "__main__":
    node_iter = ast.NodeVisitor()
    node_iter.visit_Import = visit_Import
    node_iter.visit_ImportFrom = visit_ImportFrom

    with open("/home/andi/Memgraph/code/mage/python/nxalg.py") as f:
        print("File imported")
        node_iter.visit(ast.parse(f.read()))
        print(f"Modules: {modules}")
        remove_custom_module_and_packages_from_cache(modules)
