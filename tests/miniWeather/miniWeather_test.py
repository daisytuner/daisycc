# Copyright 2023 Lukas Truemper. All rights reserved.
import subprocess
import shutil

from pathlib import Path


def test_miniWeather():
    daisycache = Path() / ".daisycache"
    shutil.rmtree(daisycache, ignore_errors=True)

    source_path = Path(__file__).parent
    source_file = Path(__file__).parent / "miniWeather.cpp"
    binary_path = Path() / "daisy_test"

    cmd = [
        "mpicc",
        "-showme:compile",
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    process.wait()
    mpi_compile_flags, stderr = process.communicate()
    cmd = [
        "mpicc",
        "-showme:link",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    process.wait()
    mpi_link_flags, stderr = process.communicate()

    cmd = (
        [
            "daisycc++",
            "-std=c++11",
            "-fopenmp",
            "-ffast-math",
        ]
        + mpi_compile_flags.strip().split()
        + [
            "-DNO_INFORM",
            "-D_NX=100",
            "-D_NZ=50",
            "-D_SIM_TIME=400",
            "-D_OUT_FREQ=400",
            "-D_DATA_SPEC=DATA_SPEC_THERMAL",
            "-I${OLCF_PARALLEL_NETCDF_ROOT}/include",
            f"{source_file}",
        ]
        + mpi_link_flags.strip().split()
        + ["-L${OLCF_PARALLEL_NETCDF_ROOT}/lib", "-lpnetcdf", "-o", "daisy_test"]
    )
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    process.wait()
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    sdfgs = [Path(path) for path in daisycache.glob("sdfg_*") if path.is_dir()]
    assert len(sdfgs) == 4

    check_output = source_path / "check_output.sh"
    cmd = [f"{check_output}", f"./daisy_test", "1e-13", "4.5e-5"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0
