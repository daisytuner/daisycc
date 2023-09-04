# Copyright 2023 Lukas Truemper. All rights reserved.
import pytest
import shutil
import numpy as np
import subprocess

from pathlib import Path

POLYBENCH_SOURCE = Path(__file__).parent / "polybench.c"


def compile_benchmark(
    source_path, out_path, size: str, dtype: str, fast_math: bool = False
) -> None:
    cmd = [
        "clang",
        "-O2",
        "-DPOLYBENCH_TIME",
        "-DPOLYBENCH_DUMP_ARRAYS",
        f"-D{size}",
        f"-D{dtype}",
        "-o",
        out_path,
        str(POLYBENCH_SOURCE),
        source_path,
        f"-I{Path(__file__).parent}",
        "-lm",
    ]
    if fast_math:
        cmd.insert(2, "-ffast-math")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode > 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0
    assert out_path.is_file()


def lift_sdfg(
    source_path, out_path, size: str, dtype: str, schedule: str, fast_math: bool = False
) -> None:
    cmd = [
        "daisycc",
        "-O2",
        f"-fschedule={schedule}",
        "-fno-unroll-loops",
        "-DPOLYBENCH_TIME",
        "-DPOLYBENCH_DUMP_ARRAYS",
        f"-D{size}",
        f"-D{dtype}",
        "-o",
        out_path,
        str(POLYBENCH_SOURCE),
        source_path,
        f"-I{Path(__file__).parent}",
        "-lm",
    ]
    if fast_math:
        cmd.insert(2, "-ffast-math")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if True:
        print(stdout)
        print(stderr)
    assert process.returncode == 0
    assert out_path.is_file()

    sdfgs = [Path(path) for path in source_path.parent.glob("sdfg_*") if path.is_dir()]
    return sdfgs


def run_benchmark(out_path, dtype: str) -> float:
    cmd = [f"{out_path}"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    runtime = float(stdout)

    data = stderr.splitlines()
    header = data.pop(0)
    assert header == "==BEGIN DUMP_ARRAYS=="

    if dtype == "DATA_TYPE_IS_INT":
        dtype = np.int32
    elif dtype == "DATA_TYPE_IS_FLOAT":
        dtype = np.float32
    else:
        dtype = np.float64

    arrays = {}
    current_array = None
    for line in data:
        if line == "==END   DUMP_ARRAYS==":
            break

        if line.startswith("begin"):
            assert current_array is None
            current_array = line[line.find(":") + 1 :].strip()
            arrays[current_array] = []
        elif line.startswith("end"):
            assert (
                current_array is not None
                and current_array == line[line.find(":") + 1 :].strip()
            )
            arrays[current_array] = np.hstack(arrays[current_array])
            current_array = None
        else:
            assert current_array is not None
            row = np.fromstring(line, dtype=dtype, sep=" ")
            arrays[current_array].append(row)

    return runtime, arrays


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_2mm(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "2mm"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_3mm(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "3mm"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


# @pytest.mark.parametrize(
#     "size, dtype, schedule",
#     [
#
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore")
#     ],
# )
# def test_adi(size, dtype, schedule):
#     benchmark_path = Path(__file__).parent / "adi"
#     source_path = benchmark_path / f"{benchmark_path.name}.c"
#     out_path = benchmark_path / f"{benchmark_path.name}.out"
#     out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

#     # Build reference
#     compile_benchmark(source_path, out_path, size, dtype)

#     # SDFG lifting
#     sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

#     # Execute reference
#     reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

#     # Execute opt
#     opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

#     with open(
#         Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
#     ) as handle:
#         lines = handle.readlines()
#         start = None
#         stop = None
#         for i, line in enumerate(lines):
#             if "tail call void (...) @polybench_timer_start()" in line:
#                 start = i
#             elif "tail call void (...) @polybench_timer_stop()" in line:
#                 assert start is not None
#                 stop = i
#                 break

#         assert start is not None and stop is not None

#         init = False
#         inserted_sdfgs = 0
#         for i in range(start + 1, stop, 1):
#             if "@__dace_init" in lines[i]:
#                 assert not init
#                 init = True
#             elif "@__dace_exit" in lines[i]:
#                 assert init
#                 init = False
#                 inserted_sdfgs += 1

#         assert inserted_sdfgs > 0

#     for array in reference_arrays:
#         assert array in opt_arrays
#         assert np.allclose(
#             reference_arrays[array],
#             opt_arrays[array],
#             atol=1e-4,
#
#             equal_nan=False,
#         )

#     shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_atax(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "atax"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_bicg(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "bicg"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_cholesky(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "cholesky"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype, fast_math=True)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule, fast_math=True)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-2,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_correlation(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "correlation"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype, fast_math=True)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule, fast_math=True)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_covariance(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "covariance"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


# @pytest.mark.parametrize(
#     "size, dtype, schedule",
#     [
#       pytest.param("SMALL_DATASET", "DATA_TYPE_IS_FLOAT", "sequential"),
#       pytest.param("SMALL_DATASET", "DATA_TYPE_IS_FLOAT", "multicore")",
#     ],
# )
# def test_deriche(size, dtype, schedule):
#     benchmark_path = Path(__file__).parent / "deriche"
#     source_path = benchmark_path / f"{benchmark_path.name}.c"
#     out_path = benchmark_path / f"{benchmark_path.name}.out"
#     out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

#     # Build reference
#     compile_benchmark(source_path, out_path, size, dtype, fast_math=True)

#     # SDFG lifting
#     sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule, fast_math=True)

#     # Execute reference
#     reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

#     # Execute opt
#     opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

#     with open(
#         Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
#     ) as handle:
#         lines = handle.readlines()
#         start = None
#         stop = None
#         for i, line in enumerate(lines):
#             if "tail call void (...) @polybench_timer_start()" in line:
#                 start = i
#             elif "tail call void (...) @polybench_timer_stop()" in line:
#                 assert start is not None
#                 stop = i
#                 break

#         assert start is not None and stop is not None

#         init = False
#         inserted_sdfgs = 0
#         for i in range(start + 1, stop + 1, 1):
#             if "@__dace_init" in lines[i]:
#                 assert not init
#                 init = True
#             elif "@__dace_exit" in lines[i]:
#                 assert init
#                 init = False
#                 inserted_sdfgs += 1

#         assert inserted_sdfgs > 0

#     for array in reference_arrays:
#         assert array in opt_arrays
#         assert np.allclose(reference_arrays[array], opt_arrays[array], atol=10 equal_nan=False)

#     shutil.rmtree(Path() / ".daisycache")


# @pytest.mark.parametrize(
#     "size, dtype, schedule",
#     [
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
#     ],
# )
# def test_doitgen(size, dtype, schedule):
#     benchmark_path = Path(__file__).parent / "doitgen"
#     source_path = benchmark_path / f"{benchmark_path.name}.c"
#     out_path = benchmark_path / f"{benchmark_path.name}.out"
#     out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

#     # Build reference
#     compile_benchmark(source_path, out_path, size, dtype)

#     # SDFG lifting
#     sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

#     # Execute reference
#     reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

#     # Execute opt
#     opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

#     with open(
#         Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
#     ) as handle:
#         lines = handle.readlines()
#         start = None
#         stop = None
#         for i, line in enumerate(lines):
#             if "tail call void (...) @polybench_timer_start()" in line:
#                 start = i
#             elif "tail call void (...) @polybench_timer_stop()" in line:
#                 assert start is not None
#                 stop = i
#                 break

#         assert start is not None and stop is not None

#         init = False
#         inserted_sdfgs = 0
#         for i in range(start + 1, stop, 1):
#             if "@__dace_init" in lines[i]:
#                 assert not init
#                 init = True
#             elif "@__dace_exit" in lines[i]:
#                 assert init
#                 init = False
#                 inserted_sdfgs += 1

#         assert inserted_sdfgs > 0

#     for array in reference_arrays:
#         assert array in opt_arrays
#         assert np.allclose(
#             reference_arrays[array],
#             opt_arrays[array],
#             atol=1e-4,
#             equal_nan=False,
#         )

#     shutil.rmtree(Path() / ".daisycache")


# @pytest.mark.parametrize(
#     "size, dtype, schedule",
#     [
#       pytest.param("SMALL_DATASET "DATA_TYPE_IS_FLOAT", "sequential"),
#       pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
#       pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore")",
#     ],
# )
# def test_durbin(size, dtype, schedule):
#     benchmark_path = Path(__file__).parent / "durbin"
#     source_path = benchmark_path / f"{benchmark_path.name}.c"
#     out_path = benchmark_path / f"{benchmark_path.name}.out"
#     out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

#     # Build reference
#     compile_benchmark(source_path, out_path, size, dtype)

#     # SDFG lifting
#     sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

#     # Execute reference
#     reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

#     # Execute opt
#     opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

#     with open(
#         Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
#     ) as handle:
#         lines = handle.readlines()
#         start = None
#         stop = None
#         for i, line in enumerate(lines):
#             if "tail call void (...) @polybench_timer_start()" in line:
#                 start = i
#             elif "tail call void (...) @polybench_timer_stop()" in line:
#                 assert start is not None
#                 stop = i
#                 break

#         assert start is not None and stop is not None

#         init = False
#         inserted_sdfgs = 0
#         for i in range(start + 1, stop, 1):
#             if "@__dace_init" in lines[i]:
#                 assert not init
#                 init = True
#             elif "@__dace_exit" in lines[i]:
#                 assert init
#                 init = False
#                 inserted_sdfgs += 1

#         assert inserted_sdfgs > 0

#     for array in reference_arrays:
#         assert array in opt_arrays
#         assert np.allclose(reference_arrays[array], opt_arrays[array], atol=10 equal_nan=False)

#     shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_fdtd_2d(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "fdtd-2d"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_INT", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_INT", "multicore"),
    ],
)
def test_floyd_warshall(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "floyd-warshall"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_gemm(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "gemm"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_gemver(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "gemver"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_gesummv(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "gesummv"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


# @pytest.mark.parametrize(
#     "size, dtype, schedule",
#     [
#
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
#     ],
# )
# def test_gramschmidt(size, dtype, schedule):
#     benchmark_path = Path(__file__).parent / "gramschmidt"
#     source_path = benchmark_path / f"{benchmark_path.name}.c"
#     out_path = benchmark_path / f"{benchmark_path.name}.out"
#     out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

#     # Build reference
#     compile_benchmark(source_path, out_path, size, dtype, fast_math=True)

#     # SDFG lifting
#     sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule, fast_math=True)

#     # Execute reference
#     reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

#     # Execute opt
#     opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

#     with open(
#         Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
#     ) as handle:
#         lines = handle.readlines()
#         start = None
#         stop = None
#         for i, line in enumerate(lines):
#             if "tail call void (...) @polybench_timer_start()" in line:
#                 start = i
#             elif "tail call void (...) @polybench_timer_stop()" in line:
#                 assert start is not None
#                 stop = i
#                 break

#         assert start is not None and stop is not None

#         init = False
#         inserted_sdfgs = 0
#         for i in range(start + 1, stop, 1):
#             if "@__dace_init" in lines[i]:
#                 assert not init
#                 init = True
#             elif "@__dace_exit" in lines[i]:
#                 assert init
#                 init = False
#                 inserted_sdfgs += 1

#         assert inserted_sdfgs > 0

#     for array in reference_arrays:
#         assert array in opt_arrays
#         assert np.allclose(
#             reference_arrays[array],
#             opt_arrays[array],
#             atol=1e-4,
#             equal_nan=False,
#         )

#     shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_heat_3d(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "heat-3d"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(reference_arrays[array], opt_arrays[array], equal_nan=True)


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_jacobi_1d(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "jacobi-1d"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_jacobi_2d(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "jacobi-2d"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_lu(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "lu"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


# @pytest.mark.parametrize(
#     "size, dtype, schedule",
#     [
#
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore")
#     ],
# )
# def test_ludcmp(size, dtype, schedule):
#     benchmark_path = Path(__file__).parent / "ludcmp"
#     source_path = benchmark_path / f"{benchmark_path.name}.c"
#     out_path = benchmark_path / f"{benchmark_path.name}.out"
#     out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

#     # Build reference
#     compile_benchmark(source_path, out_path, size, dtype)

#     # SDFG lifting
#     sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

#     # Execute reference
#     reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

#     # Execute opt
#     opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

#     with open(
#         Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
#     ) as handle:
#         lines = handle.readlines()
#         start = None
#         stop = None
#         for i, line in enumerate(lines):
#             if "tail call void (...) @polybench_timer_start()" in line:
#                 start = i
#             elif "tail call void (...) @polybench_timer_stop()" in line:
#                 assert start is not None
#                 stop = i
#                 break

#         assert start is not None and stop is not None

#         init = False
#         inserted_sdfgs = 0
#         for i in range(start + 1, stop, 1):
#             if "@__dace_init" in lines[i]:
#                 assert not init
#                 init = True
#             elif "@__dace_exit" in lines[i]:
#                 assert init
#                 init = False
#                 inserted_sdfgs += 1

#         assert inserted_sdfgs > 0

#     for array in reference_arrays:
#         assert array in opt_arrays
#         assert np.allclose(
#             reference_arrays[array],
#             opt_arrays[array],
#             atol=1e-4,
#             equal_nan=False,
#         )

#     shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_min_plus_mm(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "min_plus_mm"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype, fast_math=True)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule, fast_math=True)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_mvt(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "mvt"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_INT", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_INT", "multicore"),
    ],
)
def test_nussinov(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "nussinov"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


# @pytest.mark.parametrize(
#     "size, dtype, schedule",
#     [
#
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore")
#     ],
# )
# def test_seidel_2d(size, dtype, schedule):
#     benchmark_path = Path(__file__).parent / "seidel-2d"
#     source_path = benchmark_path / f"{benchmark_path.name}.c"
#     out_path = benchmark_path / f"{benchmark_path.name}.out"
#     out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

#     # Build reference
#     compile_benchmark(source_path, out_path, size, dtype)

#     # SDFG lifting
#     sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

#     # Execute reference
#     reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

#     # Execute opt
#     opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

#     with open(
#         Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
#     ) as handle:
#         lines = handle.readlines()
#         start = None
#         stop = None
#         for i, line in enumerate(lines):
#             if "tail call void (...) @polybench_timer_start()" in line:
#                 start = i
#             elif "tail call void (...) @polybench_timer_stop()" in line:
#                 assert start is not None
#                 stop = i
#                 break

#         assert start is not None and stop is not None

#         init = False
#         inserted_sdfgs = 0
#         for i in range(start + 1, stop, 1):
#             if "@__dace_init" in lines[i]:
#                 assert not init
#                 init = True
#             elif "@__dace_exit" in lines[i]:
#                 assert init
#                 init = False
#                 inserted_sdfgs += 1

#         assert inserted_sdfgs > 0

#     for array in reference_arrays:
#         assert array in opt_arrays
#         assert np.allclose(
#             reference_arrays[array],
#             opt_arrays[array],
#             atol=1e-4,
#             equal_nan=False,
#         )

#     shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_softmax(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "softmax"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype, fast_math=True)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule, fast_math=True)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_symm(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "symm"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_syr2k(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "syr2k"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_syrk(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "syrk"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")


# @pytest.mark.parametrize(
#     "size, dtype, schedule",
#     [
#
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
#         pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore")
#     ],
# )
# def test_trisolv(size, dtype, schedule):
#     benchmark_path = Path(__file__).parent / "trisolv"
#     source_path = benchmark_path / f"{benchmark_path.name}.c"
#     out_path = benchmark_path / f"{benchmark_path.name}.out"
#     out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

#     # Build reference
#     compile_benchmark(source_path, out_path, size, dtype)

#     # SDFG lifting
#     sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

#     # Execute reference
#     reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

#     # Execute opt
#     opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

#     with open(
#         Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
#     ) as handle:
#         lines = handle.readlines()
#         start = None
#         stop = None
#         for i, line in enumerate(lines):
#             if "tail call void (...) @polybench_timer_start()" in line:
#                 start = i
#             elif "tail call void (...) @polybench_timer_stop()" in line:
#                 assert start is not None
#                 stop = i
#                 break

#         assert start is not None and stop is not None

#         init = False
#         inserted_sdfgs = 0
#         for i in range(start + 1, stop, 1):
#             if "@__dace_init" in lines[i]:
#                 assert not init
#                 init = True
#             elif "@__dace_exit" in lines[i]:
#                 assert init
#                 init = False
#                 inserted_sdfgs += 1

#         assert inserted_sdfgs > 0

#     for array in reference_arrays:
#         assert array in opt_arrays
#         assert np.allclose(
#             reference_arrays[array],
#             opt_arrays[array],
#             atol=1e-4,
#             equal_nan=False,
#         )

#     shutil.rmtree(Path() / ".daisycache")


@pytest.mark.parametrize(
    "size, dtype, schedule",
    [
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "sequential"),
        pytest.param("SMALL_DATASET", "DATA_TYPE_IS_DOUBLE", "multicore"),
    ],
)
def test_trmm(size, dtype, schedule):
    benchmark_path = Path(__file__).parent / "trmm"
    source_path = benchmark_path / f"{benchmark_path.name}.c"
    out_path = benchmark_path / f"{benchmark_path.name}.out"
    out_opt_path = benchmark_path / f"{benchmark_path.name}_daisy.out"

    # Build reference
    compile_benchmark(source_path, out_path, size, dtype)

    # SDFG lifting
    sdfgs = lift_sdfg(source_path, out_opt_path, size, dtype, schedule)

    # Execute reference
    reference_runtime, reference_arrays = run_benchmark(out_path, dtype)

    # Execute opt
    opt_runtime, opt_arrays = run_benchmark(out_opt_path, dtype)

    with open(
        Path() / ".daisycache" / f"{out_opt_path.stem}.ll", mode="r", encoding="utf-8"
    ) as handle:
        lines = handle.readlines()
        start = None
        stop = None
        for i, line in enumerate(lines):
            if "tail call void (...) @polybench_timer_start()" in line:
                start = i
            elif "tail call void (...) @polybench_timer_stop()" in line:
                assert start is not None
                stop = i
                break

        assert start is not None and stop is not None

        init = False
        inserted_sdfgs = 0
        for i in range(start + 1, stop, 1):
            if "@__dace_init" in lines[i]:
                assert not init
                init = True
            elif "@__dace_exit" in lines[i]:
                assert init
                init = False
                inserted_sdfgs += 1

        assert inserted_sdfgs > 0

    for array in reference_arrays:
        assert array in opt_arrays
        assert np.allclose(
            reference_arrays[array],
            opt_arrays[array],
            atol=1e-4,
            equal_nan=False,
        )

    shutil.rmtree(Path() / ".daisycache")
