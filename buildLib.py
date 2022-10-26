import os


def run(command):
    print(f"\u001b[34m{command}\u001b[0m")
    if os.system(command) != 0:
        print("\u001b[31mERROR IN COMPILATION\u001b[0m")
        exit(-1)


def run_compilation(files, f):
    for src, dst in files:
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            print(f"\u001b[32mCompiling {src}\u001b[0m")
            command = f(src, dst)
            run(command)
        else:
            print(f"\u001b[32mSkipping {src}\u001b[0m")


def buildLib(ccs, nvcc, cxx="g++", verbose=True):
    include_dirs = ["include"]
    cu_files = [('src/fpj.cu', 'objs/fpj.o'), ('src/fbp.cu', 'objs/fbp.o')]
    all_objects = [y for _, y in cu_files]
    include_flags = [f"-I{x}" for x in include_dirs]

    nvcc_flags = ["-std=c++11", f"-ccbin={cxx}", "-Xcompiler", "-fPIC", "-Xcompiler -static",
                  "-Xcompiler -static-libgcc", "-Xcompiler -static-libstdc++"] + include_flags + \
        [f"-gencode arch=compute_{x},code=sm_{x}" for x in ccs] + [
        "-DNDEBUG -O3 --generate-line-info --compiler-options -Wall"]

    if verbose:
        nvcc_flags.append("-DVERBOSE")

    nvcc_flags = " ".join(nvcc_flags)

    # compile
    if not os.path.exists('objs'):
        os.makedirs('objs')
    run_compilation(cu_files, lambda src, dst: f"{nvcc} {nvcc_flags} -c {src} -o {dst}")

    run(f"ar rc objs/libmango.a {' '.join(all_objects)}")
