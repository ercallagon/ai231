import py_compile, sys
try:
    py_compile.compile("/mnt/jfs_hpc/home/edgardo.callagon.jr/sandbox/tmp/build_notebook.py", doraise=True)
    print("COMPILE_OK")
except py_compile.PyCompileError as e:
    print("COMPILE_FAIL")
    print(e)
