import dace

from scop2sdfg.scop.types import TYPES


def test_table():
    assert any(map(lambda t: isinstance(t, str), TYPES.keys()))
    assert any(map(lambda t: isinstance(t, dace.typeclass), TYPES.values()))


def test_mapping():
    assert TYPES["i1"] == dace.bool
    assert TYPES["i8"] == dace.int8
    assert TYPES["i16"] == dace.int16
    assert TYPES["i32"] == dace.int32
    assert TYPES["i64"] == dace.int64
    assert TYPES["float"] == dace.float32
    assert TYPES["double"] == dace.float64
