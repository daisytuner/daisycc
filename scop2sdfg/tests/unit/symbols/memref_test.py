import dace

from scop2sdfg.scop.symbols.memref import Memref


def test_value():
    desc = {
        "kind": "value",
        "name": "MemRef2",
        "sizes": [],
        "type": "double",
        "variable": "double %4",
    }
    memref = Memref.from_json(desc)

    assert memref.reference == "%4"
    assert memref.dtype == dace.float64

    assert memref.name == "MemRef2"
    assert memref.kind == "value"

    assert len(memref.shape) == 0


def test_1d_array():
    desc = {
        "kind": "array",
        "name": "MemRef2",
        "sizes": ["32"],
        "type": "double",
        "variable": "  %6 = tail call ptr @polybench_alloc_data(i64 noundef 1080000, i32 noundef 8) #6",
    }
    memref = Memref.from_json(desc)

    assert memref.reference == "%6"
    assert memref.dtype == dace.float64

    assert memref.name == "MemRef2"
    assert memref.kind == "array"

    assert len(memref.shape) == 1
    assert memref.shape[0] == 32


def test_1d_array_symbolic():
    desc = {
        "kind": "array",
        "name": "MemRef2",
        "sizes": ["*"],
        "type": "double",
        "variable": "  %6 = tail call ptr @polybench_alloc_data(i64 noundef 1080000, i32 noundef 8) #6",
    }
    memref = Memref.from_json(desc)

    assert memref.reference == "%6"
    assert memref.dtype == dace.float64

    assert memref.name == "MemRef2"
    assert memref.kind == "array"

    assert len(memref.shape) == 1
    assert isinstance(memref.shape[0], dace.symbolic.symbol)


def test_2d_array():
    desc = {
        "kind": "array",
        "name": "MemRef2",
        "sizes": ["*", "1200"],
        "type": "double",
        "variable": "  %6 = tail call ptr @polybench_alloc_data(i64 noundef 1080000, i32 noundef 8) #6",
    }
    memref = Memref.from_json(desc)

    assert memref.reference == "%6"
    assert memref.dtype == dace.float64

    assert memref.name == "MemRef2"
    assert memref.kind == "array"

    assert len(memref.shape) == 2
    assert isinstance(memref.shape[0], dace.symbolic.symbol)
    assert memref.shape[1] == 1200
