import os
import copy
import sys
import shutil
import argparse
import subprocess

from typing import List
from pathlib import Path

LIB_DAISY = Path(os.environ["CONDA_PREFIX"]) / "lib" / "libDaisyLLVMPlugin.so"


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


def main():
    argv = sys.argv
    executable = Path(argv[0]).name
    if executable == "daisycc":
        compiler = "clang"
    else:
        compiler = "clang++"

    parser = argparse.ArgumentParser(
        prog="daisycc",
        description="Daisy Optimizing C/C++ Compilers based on LLVM and DaCe",
    )
    parser.add_argument("--version", action="store_true", default=False)
    parser.add_argument("-v", action="store_true", default=False)

    # Input and output files
    parser.add_argument("inputs", type=str, nargs="*")
    parser.add_argument("-o", default="a.out")

    # Macros
    parser.add_argument("-D", action="append", default=[])
    parser.add_argument("-U", action="append", default=[])

    # Includes
    parser.add_argument("-isystem", action="append", default=[])
    parser.add_argument("-I", action="append", default=[])

    # Linking
    parser.add_argument("-L", action="append", default=[])
    parser.add_argument("-l", action="append", default=[])

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
        _execute_command(f"{compiler}")
    elif args.version:
        _execute_command(f"{compiler} --version")
    elif args.v:
        _execute_command(f"{compiler} -v")
    else:
        # Compile
        output_file = Path(args.o)

        ## Cache folder
        cache_folder = Path() / ".daisycache"
        if cache_folder.is_dir():
            shutil.rmtree(cache_folder)
        cache_folder.mkdir(exist_ok=False, parents=False)

        # Compile and lift
        llvm_base_command = [
            compiler,
            "-S",
            "-emit-llvm",
            "-O2",
            "--gcc-toolchain=/usr",
        ]
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
        includes = ["-isystem" + arg for arg in args.isystem] + [
            "-I" + arg for arg in args.I
        ]
        llvm_command = llvm_base_command + compile_options + macros + includes

        llvm_source_files = []
        sdfg_libs = []
        for input_file in (Path(file) for file in args.inputs):
            llvm_file = str(cache_folder / f"{input_file.stem}.ll")
            llvm_file_command = copy.copy(llvm_command) + [input_file, "-o", llvm_file]
            ret_code = _execute_command(llvm_file_command)
            if ret_code > 0:
                return ret_code

            llvm_file_lifted = cache_folder / f"{input_file.stem}_lifted.ll"
            plugin = [
                "opt",
                "-S",
                f"--load-pass-plugin={LIB_DAISY}",
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

            name = input_file.name.replace(".", "").replace("-", "_")
            sdfg_libs.extend(
                [Path(path) for path in cache_folder.glob(f"libsdfg_{name}_*.so")]
            )

        # Link LLVM files
        linker_comand = [
            "llvm-link",
            "-S",
            "-o",
            str(cache_folder / f"{output_file.stem}.ll"),
        ] + llvm_source_files
        ret_code = _execute_command(linker_comand)
        if ret_code > 0:
            return ret_code

        # Assemble LLVM files
        llc_command = ["llc", "-filetype=obj", "-O2"]
        if args.fPIE:
            llc_command.append("-relocation-model=pic")

        llc_command += ["-o", str(cache_folder / f"{output_file.stem}.o")]
        llc_command += [str(cache_folder / f"{output_file.stem}.ll")]
        ret_code = _execute_command(llc_command)
        if ret_code > 0:
            return ret_code

        # Build
        build_command = [
            compiler,
            "--gcc-toolchain=/usr",
            str(cache_folder / f"{output_file.stem}.o"),
        ]
        build_command += ["-o", output_file]

        build_command += ["-L" + arg for arg in args.L]
        build_command.append(f"-L{cache_folder.absolute()}")

        build_command += ["-l" + arg for arg in args.l]
        for sdfg_lib in sdfg_libs:
            build_command.append(f"-l{sdfg_lib.stem[3:]}")

        build_command.append(f"-Wl,-rpath={cache_folder.absolute()}")

        ret_code = _execute_command(build_command)
        if ret_code > 0:
            return ret_code
