import os
import copy
import sys
import shutil
import argparse
import subprocess

from typing import List
from pathlib import Path


def find_plugin():
    if "LIBDAISY_PATH" in os.environ:
        return os.environ["LIBDAISY_PATH"]

    with subprocess.Popen(
        ["ldconfig", "-p"],
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
    ) as p:
        res = str(p.stdout.read())
        found = res.find("libDaisyLLVMPlugin.so")
        assert (
            found != -1
        ), "Could not find libDaisyLLVMPlugin.so. Please install the plugin and add it to your library search paths (ldconfig)"
        res = res[found:]
        res = res[: res.find("\\n")]
        path = res.split("=>")[1].strip()
        return path


def _execute_command(command: List[str]):
    process = subprocess.Popen(
        command,
        shell=False,
        stdout=subprocess.PIPE,
    )
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break

        if output:
            out_str = output.__str__()
            print(out_str)
    return process.returncode


def _compile(compiler, args, output_file, cache_folder, plugin_path):
    ## Opt level
    if args.O3:
        opt_level = "-O3"
    elif args.O2:
        opt_level = "-O2"
    elif args.O1:
        opt_level = "-O1"
    elif args.O0:
        opt_level = "-O0"

    # Compile and lift
    llvm_base_command = [
        compiler,
        "-S",
        "-emit-llvm",
        opt_level,
    ]
    if args.std is not None:
        llvm_base_command.append("-std=" + args.std)
    if args.stdlib is not None:
        llvm_base_command.append("-stdlib=" + args.stdlib)
    if args.target is not None:
        llvm_base_command.append("--target=" + args.target)
    if args.sysroot is not None:
        llvm_base_command.append("--sysroot=" + args.sysroot)
    if args.v:
        llvm_base_command.append("-v")
    if args.w:
        llvm_base_command.append("-w")
    compile_options = [
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-fno-tree-vectorize",
    ]
    if args.ffast_math:
        compile_options.append("-ffast-math")
    if args.fno_unroll_loops:
        compile_options.append("-fno-unroll-loops")
    macros = ["-D" + arg for arg in args.D] + ["-U" + arg for arg in args.U]
    wanings = ["-W" + arg for arg in args.W]
    includes = ["-isystem" + arg for arg in args.isystem] + [
        "-I" + arg for arg in args.I
    ]
    llvm_command = llvm_base_command + compile_options + macros + wanings + includes

    llvm_source_files = []
    for input_file in (Path(file) for file in args.inputs):
        llvm_file = str(cache_folder / f"{input_file.stem}.ll")
        llvm_file_command = copy.copy(llvm_command) + [input_file, "-o", llvm_file]
        ret_code = _execute_command(llvm_file_command)
        if ret_code > 0:
            return ret_code

        llvm_file_lifted = cache_folder / f"{input_file.stem}_lifted.ll"
        plugin = [
            "opt-16",
            "-S",
            f"--load-pass-plugin={plugin_path}",
            "--passes=Daisy",
            f"--daisy-schedule={args.fschedule}",
            f"--daisy-transfer-tune={args.ftransfer_tune}",
        ]
        polly = [
            "-polly-process-unprofitable",
        ]
        files = [
            llvm_file,
            "-o",
            llvm_file_lifted,
        ]
        opt_command = plugin + polly + files
        ret_code = _execute_command(opt_command)
        if ret_code > 0:
            return ret_code

        llvm_source_files.append(llvm_file_lifted)

    # Link LLVM files
    linker_comand = [
        "llvm-link-16",
        "-S",
        "-o",
        str(cache_folder / f"{output_file.stem}.ll"),
    ] + llvm_source_files
    ret_code = _execute_command(linker_comand)
    if ret_code > 0:
        return ret_code

    # Assemble LLVM files
    llc_command = ["llc-16", "-filetype=obj", opt_level]
    if args.fPIE:
        llc_command.append("-relocation-model=pic")

    llc_command += [str(cache_folder / f"{output_file.stem}.ll")]
    ret_code = _execute_command(llc_command)
    return ret_code


def _build(compiler, args, input_files, output_file, cache_folder):
    build_command = [
        compiler,
    ] + input_files
    if args.target is not None:
        build_command.append("--target=" + args.target)
    if args.sysroot is not None:
        build_command.append("--sysroot=" + args.sysroot)
    build_command += ["-o", output_file]

    build_command += ["-L" + arg for arg in args.L]
    build_command.append(f"-L{cache_folder.absolute()}")

    build_command += ["-l" + arg for arg in args.l]
    sdfg_libs = [Path(path) for path in cache_folder.glob(f"libsdfg_*.so")]
    for sdfg_lib in sdfg_libs:
        build_command.append(f"-l{sdfg_lib.stem[3:]}")

    build_command.append(f"-Wl,-rpath={cache_folder.absolute()}")

    ret_code = _execute_command(build_command)
    return ret_code


def main():
    plugin_path = find_plugin()

    argv = sys.argv
    executable = Path(argv[0]).name
    if executable == "daisycc":
        compiler = "clang-16"
    else:
        compiler = "clang++-16"

    parser = argparse.ArgumentParser(
        prog="daisycc",
        description="Daisy Optimizing C/C++ Compilers based on LLVM and DaCe",
    )
    parser.add_argument("--version", action="store_true", default=False)
    parser.add_argument("-v", action="store_true", default=False)

    # Actions
    parser.add_argument("-c", "--compile", action="store_true", default=False)

    # Input and output files
    parser.add_argument("inputs", type=str, nargs="*")
    parser.add_argument("-o", default="a.out")

    # Macros
    parser.add_argument("-D", action="append", default=[])
    parser.add_argument("-U", action="append", default=[])

    # Warnings
    parser.add_argument("-w", action="store_true", default=False)
    parser.add_argument("-W", action="append", default=[])

    # Includes
    parser.add_argument("-isystem", action="append", default=[])
    parser.add_argument("-I", action="append", default=[])

    # Linking
    parser.add_argument("-L", action="append", default=[])
    parser.add_argument("-l", action="append", default=[])

    # C++ standard
    parser.add_argument(
        "-std",
        choices=["c++98", "c++11", "c++14", "c++17", "C89", "C99", "c11", "c17", "C23"],
    )
    parser.add_argument("-stdlib", choices=["libc++", "libstdc++", "platform"])

    # Optimization levels
    parser.add_argument("-O0", action="store_true", default=False)
    parser.add_argument("-O1", action="store_true", default=False)
    parser.add_argument("-O2", action="store_true", default=True)
    parser.add_argument("-O3", action="store_true", default=False)

    # Target
    parser.add_argument("--target", type=str)
    parser.add_argument("--sysroot", type=str)

    # Compiler options
    parser.add_argument("-ffast-math", action="store_true", default=False)
    parser.add_argument("-fno-unroll-loops", action="store_true", default=False)
    parser.add_argument("-fopenmp", action="store_true", default=False)
    parser.add_argument("-fPIE", action="store_true", default=True)

    # Scheduling options
    parser.add_argument("-ftransfer-tune", action="store_true", default=False)
    parser.add_argument(
        "-fschedule",
        choices=["sequential", "multicore", "gpu"],
        default="sequential",
        help="",
    )

    # Start of Program

    args = parser.parse_args()
    if len(argv) == 1:
        _execute_command([compiler])
    elif len(argv) == 2 and args.v:
        _execute_command([compiler, "-v"])
    elif args.version:
        _execute_command([compiler, "--version"])
    else:
        # Determine input types
        input_types = map(lambda input_file: input_file.endswith(".o"), args.inputs)
        object_file_input = any(input_types)
        assert not object_file_input or all(input_types)

        output_file = Path(args.o)
        cache_folder = Path() / ".daisycache"

        if not object_file_input:
            if cache_folder.is_dir():
                shutil.rmtree(cache_folder)
            cache_folder.mkdir(exist_ok=False, parents=False)

            tmp_output = cache_folder / f"{output_file.stem}.o"
            ret_code = _compile(compiler, args, tmp_output, cache_folder, plugin_path)
            if ret_code > 0:
                return ret_code

            if args.compile:
                shutil.copy(tmp_output, output_file)
                return ret_code

            ret_code = _build(compiler, args, [tmp_output], output_file, cache_folder)
            return ret_code
        else:
            assert cache_folder.is_dir()
            ret_code = _build(compiler, args, args.inputs, output_file, cache_folder)
            return ret_code
