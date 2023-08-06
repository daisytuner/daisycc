import dace

from scop2sdfg.scop.undefined_value import UndefinedValue
from scop2sdfg.scop.symbols.constant import Constant
from scop2sdfg.scop.computation.indirection import Indirection


def test_load_1d():
    arguments = [UndefinedValue("%11", "i64")]
    indirection = Indirection(
        "%10",
        "double",
        "read",
        "  %10 = load double, ptr %9, align 8, !tbaa !9",
        "",
        "MemRef1",
        arguments,
    )
    assert indirection.reference == "%10"
    assert indirection.array == "MemRef1"
    assert indirection.dtype == dace.float64

    assert len(indirection.arguments()) == 1

    assert indirection.as_cpp() == "_in[_11]"

    array = dace.data.Array(dace.float64, shape=[256])
    memlet: dace.Memlet = indirection.memlet(array)
    assert str(memlet) == "MemRef1[0:256]"


def test_load_2d():
    arguments = [UndefinedValue("%11", "i64"), UndefinedValue("%12", "i64")]
    indirection = Indirection(
        "%10",
        "double",
        "read",
        "  %10 = load double, ptr %9, align 8, !tbaa !9",
        "",
        "MemRef1",
        arguments,
    )
    assert indirection.reference == "%10"
    assert indirection.array == "MemRef1"
    assert indirection.dtype == dace.float64

    assert len(indirection.arguments()) == 2

    assert indirection.as_cpp() == "_in[_11,_12]"

    array = dace.data.Array(dace.float64, shape=[32, 256])
    memlet: dace.Memlet = indirection.memlet(array)
    assert str(memlet) == "MemRef1[0:32, 0:256]"


def test_load_2d_with_constant():
    arguments = [Constant("ABC", "i64", "10"), UndefinedValue("%12", "i64")]
    indirection = Indirection(
        "%10",
        "double",
        "read",
        "  %10 = load double, ptr %9, align 8, !tbaa !9",
        "",
        "MemRef1",
        arguments,
    )
    assert indirection.reference == "%10"
    assert indirection.array == "MemRef1"
    assert indirection.dtype == dace.float64

    assert len(indirection.arguments()) == 2

    assert indirection.as_cpp() == "_in[_12]"

    array = dace.data.Array(dace.float64, shape=[32, 256])
    memlet: dace.Memlet = indirection.memlet(array)
    assert str(memlet) == "MemRef1[10, 0:256]"
